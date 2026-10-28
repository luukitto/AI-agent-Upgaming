---
title: Incident Report INC-0325 - Elevated false declines
date: 2026-03-19
severity: SEV-2
---
# Incident Report INC-0325: Elevated false declines

On 2026-03-19 between 14:00 and 16:05 UTC the fraud-scoring service declined about 4.1% of legitimate card payments (normal rate is 0.3%).

# Root cause
Fraud model v7.2, deployed with change CHG-2045 on 2026-03-18, used decline thresholds that were too strict for low-value payments.

# Impact
fraud-scoring and payment-gateway (payments were declined, not delayed). Around 2,900 legitimate payments were declined. Payment latency was normal.

# Resolution
The model was rolled back to v7.1 at 16:05 UTC.
