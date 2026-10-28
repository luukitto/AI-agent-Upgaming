---
title: NovaPay Platform Architecture Overview
owner: Platform Engineering
updated: 2026-02-15
---
# NovaPay Platform Architecture Overview

NovaPay processes card and bank-transfer payments for merchants in Georgia and the EU.

# Core services

The payment-gateway service is the entry point for all payment requests. It validates requests, calls fraud-scoring synchronously, stores payment state in the payments-db PostgreSQL cluster, and publishes events to the ledger-events queue.

The fraud-scoring service returns a risk score for each payment within 150 ms. Payments with a score above the threshold are declined.

The ledger-service consumes the ledger-events queue and writes double-entry records to the ledger_entries table. A daily reconciliation job runs at 02:00 UTC and compares ledger totals with the acquirer settlement files.

The notification-service consumes payment-completed events and sends receipts by e-mail and SMS.

The auth-service issues JWT tokens for merchant API clients.

# Dependencies

payment-gateway depends on payments-db, fraud-scoring and the external card acquirer Acquira. ledger-service depends on the ledger-events queue produced by payment-gateway, so any slowdown in payment-gateway delays ledger postings. notification-service depends on payment-completed events.
