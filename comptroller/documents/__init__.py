"""Realistic document generation (receipts, invoices) the workflows parse for real."""
from .receipts import GeneratedReceipt, build_sample_receipts, generate_receipt

__all__ = ["GeneratedReceipt", "build_sample_receipts", "generate_receipt"]
