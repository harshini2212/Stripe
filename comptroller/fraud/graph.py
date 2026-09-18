"""Entity-graph construction and fraud-ring detection over Stripe Issuing card activity.

We build a heterogeneous graph linking cards, devices, IPs, and merchants, then
collapse it to a *card-to-card* graph whose edges encode shared infrastructure:

* **Shared device** — always a strong link. Legit employees each have their own
  device; only compromised cards in a ring share one.
* **Cross-metro shared IP** — an IP whose cards span two or more home metros looks
  like a ring (random victims across the org), whereas a single-metro shared IP is
  ordinary office / VPN egress and is ignored. This separation is *count-independent*,
  so it holds whether a metro has 4 employees or 40.

Connected components with >=2 cards are candidate fraud rings; each is scored by the
risk profile of its transactions (shared devices, merchant risk, foreign geo,
odd-hour share).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from ..domain import Dataset
from ..domain.enums import RiskBand

# An IP shared by more than this many cards is treated as pure infrastructure.
IP_FANOUT_CAP = 12


@dataclass
class RingFinding:
    """A detected shared-infrastructure cluster of cards — a candidate fraud ring."""

    ring_id: str
    card_ids: list[str]
    employee_ids: list[str]
    device_ids: list[str]
    ip_ids: list[str]
    txn_ids: list[str]
    n_txns: int
    total_exposure_cents: int
    suspicion: float  # [0, 1]
    shared_devices: int
    shared_ips: int

    @property
    def risk_band(self) -> RiskBand:
        return RiskBand.from_score(self.suspicion)

    def to_dict(self) -> dict:
        return {
            "ring_id": self.ring_id,
            "cards": self.card_ids,
            "employees": self.employee_ids,
            "shared_devices": self.device_ids,
            "shared_ips": self.ip_ids,
            "n_txns": self.n_txns,
            "total_exposure_usd": round(self.total_exposure_cents / 100, 2),
            "suspicion": round(self.suspicion, 3),
            "risk_band": self.risk_band.value,
        }


class EntityGraph:
    """Builds and queries the card/device/IP/merchant graph for a tenant."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._merchant_index = dataset.merchant_index()
        self._employee_index = dataset.employee_index()
        self._card_to_employee = {c.id: c.employee_id for c in dataset.cards}
        self._card_to_metro = {
            c.id: self._employee_index[c.employee_id].home_geo
            for c in dataset.cards if c.employee_id in self._employee_index
        }
        self.device_cards: dict[str, set[str]] = defaultdict(set)
        self.ip_cards: dict[str, set[str]] = defaultdict(set)
        self.card_txns: dict[str, list] = defaultdict(list)
        self.G = nx.Graph()
        self._build()

    def _build(self) -> None:
        for t in self.dataset.card_transactions:
            self.card_txns[t.card_id].append(t)
            if t.device_id:
                self.device_cards[t.device_id].add(t.card_id)
            if t.ip:
                self.ip_cards[t.ip].add(t.card_id)
            self.G.add_edge(("card", t.card_id), ("merchant", t.merchant_id))
            if t.device_id:
                self.G.add_edge(("card", t.card_id), ("device", t.device_id))
            if t.ip:
                self.G.add_edge(("card", t.card_id), ("ip", t.ip))

    def _ip_is_ring_like(self, ip: str) -> bool:
        """An IP looks ring-like if its cards span >=2 home metros (random victims)."""
        cards = self.ip_cards.get(ip, set())
        if not (2 <= len(cards) <= IP_FANOUT_CAP):
            return False
        metros = {self._card_to_metro.get(c) for c in cards}
        metros.discard(None)
        return len(metros) >= 2

    # ---- card-to-card shared-infrastructure graph ----------------------------
    def _card_graph(self) -> nx.Graph:
        cg = nx.Graph()
        cg.add_nodes_from(c.id for c in self.dataset.cards)
        for device, cards in self.device_cards.items():
            cards = sorted(cards)
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    cg.add_edge(cards[i], cards[j], via=("device", device))
        for ip, cards in self.ip_cards.items():
            if not self._ip_is_ring_like(ip):
                continue
            cards = sorted(cards)
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    cg.add_edge(cards[i], cards[j], via=("ip", ip))
        return cg

    def txn_graph_features(self) -> dict[str, dict[str, float]]:
        """Per-transaction graph features (no ground-truth leakage)."""
        cg = self._card_graph()
        comp_size: dict[str, int] = {}
        for comp in nx.connected_components(cg):
            multi = len(comp) > 1
            for card in comp:
                comp_size[card] = len(comp) if multi else 1

        feats: dict[str, dict[str, float]] = {}
        for t in self.dataset.card_transactions:
            dev_fanout = len(self.device_cards.get(t.device_id, {t.card_id})) if t.device_id else 1
            ip_fanout = len(self.ip_cards.get(t.ip, {t.card_id})) if t.ip else 1
            size = comp_size.get(t.card_id, 1)
            feats[t.id] = {
                "device_card_fanout": float(dev_fanout),
                "ip_card_fanout": float(ip_fanout),
                "ip_cross_metro": 1.0 if (t.ip and self._ip_is_ring_like(t.ip)) else 0.0,
                "ring_component_size": float(size),
                "in_suspected_ring": 1.0 if size > 1 else 0.0,
            }
        return feats

    def detect_rings(self) -> list[RingFinding]:
        """Return candidate fraud rings, highest suspicion first."""
        cg = self._card_graph()
        findings: list[RingFinding] = []
        ring_n = 0
        for comp in nx.connected_components(cg):
            if len(comp) < 2:
                continue
            cards = sorted(comp)
            all_txns = [t for c in cards for t in self.card_txns.get(c, [])]
            # Scope to the actually shared-infrastructure transactions so "exposure"
            # reflects the suspicious activity, not the cards' entire legit history.
            txns = [t for t in all_txns
                    if (t.device_id and len(self.device_cards[t.device_id]) > 1)
                    or (t.ip and self._ip_is_ring_like(t.ip))] or all_txns
            if not txns:
                continue
            devices = sorted({t.device_id for t in txns
                              if t.device_id and len(self.device_cards[t.device_id]) > 1})
            ips = sorted({t.ip for t in txns if t.ip and self._ip_is_ring_like(t.ip)})
            n = len(txns)
            mrisk = sum(self._merchant_index[t.merchant_id].risk_score for t in txns) / n
            foreign = sum(1 for t in txns if self._merchant_index[t.merchant_id].country != "US") / n
            odd = sum(1 for t in txns if t.ts.hour <= 5 or t.ts.hour >= 23) / n
            cnp = sum(1 for t in txns if t.channel.value == "card_not_present") / n
            # A shared device across cards is the single strongest ring signal.
            suspicion = float(min(1.0,
                                  0.30 * (len(devices) > 0)
                                  + 0.30 * mrisk
                                  + 0.15 * foreign
                                  + 0.15 * odd
                                  + 0.10 * cnp))
            findings.append(RingFinding(
                ring_id=f"ring_cand_{ring_n:02d}",
                card_ids=cards,
                employee_ids=sorted({self._card_to_employee.get(c, "?") for c in cards}),
                device_ids=devices, ip_ids=ips,
                txn_ids=[t.id for t in txns], n_txns=n,
                total_exposure_cents=sum(t.amount_cents for t in txns),
                suspicion=suspicion, shared_devices=len(devices), shared_ips=len(ips),
            ))
            ring_n += 1
        findings.sort(key=lambda f: f.suspicion, reverse=True)
        return findings
