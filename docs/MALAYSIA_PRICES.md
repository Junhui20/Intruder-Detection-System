# Malaysia pricing: cloud camera AI subscriptions vs. local hardware

Researched 2026-09-12. Exchange rate: **1 USD = 4.07 MYR** (xe.com mid-market rate, 4.06997930, checked 2026-09-12, https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=USD&To=MYR).

## Table A — Cloud subscriptions ("familiar faces" / AI captions)

| Service | Plan | Includes faces? AI captions? | MYR/month | MYR/year | Source |
|---|---|---|---|---|---|
| **Arlo Secure** — not sold in MY (Malaysia absent from Arlo's own 39-territory list) | Secure Plus, Single cam | Facial recognition, named vehicle alerts (full Arlo Intelligence AI) | $7.99 → RM32.52 (USD equiv.) | $95.88 → RM390.24 | kb.arlo.com/000062249 (availability, 2026-09-12); subger.com/en/service/arlo-secure (price, 2026-09-12) |
| Arlo Secure | Secure Plus, Unlimited cams | Same as above, all cameras | $17.99 → RM73.22 | $215.88 → RM878.62 | same |
| Arlo Secure | Secure Premium (+ pro monitoring) | Same + 24/7 monitoring | $24.99 → RM101.71 | $299.88 → RM1,220.51 | same |
| **TP-Link Tapo Care** — sold in MY (Basic plan confirmed live in Malaysia), but MY RM price page is app/portal-only (JS) | Basic ("Cloud") | AI object detection only, no faces | not found — MY portal is JS-only; global ref $3.49 → RM14.20 | $34.99 → RM142.41 | tp-link.com/my/tapocare (plan text, 2026-09-12); tapo.com/us/tapocare (global USD ref, 2026-09-12) |
| Tapo Care | Cloud Advanced AI ("Aireal") | **Facial recognition ("familiar faces"), AI Chat, Intelligent Summaries** | not found for MY; global ref $19.99 → RM81.36 | $199.99 → RM813.96 | tapo.com/us/tapocare, 2026-09-12 |
| **Imou Protect** — sold in MY (official Imou Malaysia Shopee store exists), no static MY RM price page found | Basic (7-day) | AI object detection (person/pet/vehicle/package/sound) — **no face recognition in Protect plans** | not found for MY; EU ref €3.49 → RM~15.5 | not found | store.imou.com/en-uk (EU price, 2026-09-12); imou.com/my (no price shown), 2026-09-12 |
| **EZVIZ CloudPlay** — sold in MY, RM pricing confirmed | Individual, 7-day history | Video history only + free local human/vehicle AI alerts — **no face recognition in CloudPlay** | RM16.99 | RM169.99 | ezviz.com/my/cloudplay, 2026-09-12 |
| EZVIZ CloudPlay | Individual, 30-day history | same | RM28.99 | RM289.99 | same |
| EZVIZ CloudPlay | Home (4 cams), 7-day | same | RM25.49 | RM254.99 | same |
| EZVIZ CloudPlay | Home (4 cams), 30-day | same | RM43.49 | RM434.99 | same |
| **Ring Protect** — not sold in MY (no MY store/support region; Ring not officially retailed here) | Solo | Basic person alerts, no faces | $4.99 → RM20.31 | $49.99 → RM203.46 | ring.com/protect-plans, 2026-09-12 |
| Ring Protect | Multi | same, all cameras | $9.99 → RM40.66 | $99.99 → RM406.96 | same |
| Ring Protect | Pro | **Familiar Faces (facial recognition, named alerts)** | $19.99 → RM81.36 | $199.99 → RM813.96 | same |
| **Google Home Premium / Nest Aware** — not sold in MY (APAC: only Australia, Japan, NZ) | Standard | 30-day history, smart alerts, no AI descriptions | $10 → RM40.70 | $100 → RM407.00 | store.google.com/product/google_home_premium; APAC country list via search, 2026-09-12 |
| Google Home Premium | Advanced | **AI event descriptions (Gemini), Home Brief summaries**, 60-day history | $20 → RM81.40 | $200 → RM814.00 | same |

## Table B — One-time local hardware

| Item | Typical MYR | Concrete listing | Note |
|---|---|---|---|
| Intel N100 mini PC, 8–16GB RAM | RM800–RM2,850 (wide spread; budget DDR5 boards RM800–1,000, branded 16GB units RM2,500+) | "GK3V Plus" N100, 8GB/16GB RAM: RM1,425–RM2,477 (Shopee, seller Dragon Mall) | Seen via my.biggo.com/s/N100%20mini%20pc/, 2026-09-12. Shopee/Lazada MY product pages are JS-only SPAs and returned no readable content directly — BigGo Malaysia (server-rendered price aggregator) used as next-best source, reflecting live Shopee/Lazada listings. |
| Raspberry Pi 5, 8GB RAM | ~RM782 (board only, official distributor) | RM781.70 (RM775.45 at 30+ units) | Cytron Technologies Malaysia, official Raspberry Pi reseller: my.cytron.io/p-raspberry-pi-5-with-8gb-ram, fetched 2026-09-12. |

## What a year costs

No subscription with **confirmed MY ringgit pricing** in this survey includes facial recognition / familiar faces — EZVIZ CloudPlay (the only service with a directly-readable MY price page) is storage-only, and Tapo Care's face-recognition tier ("Cloud Advanced AI") is sold in Malaysia but its RM price could not be read (app/portal-only). Taking the global USD reference for that tier as the closest comparison:

- Cheapest face-recognition-capable cloud plan (Tapo Care Cloud Advanced AI, global USD ref): **≈ RM814/year** ($199.99).
- One-time Raspberry Pi 5 8GB (Cytron MY, official): **RM781.70** — cheaper than *one year* of that subscription, and free every year after.
- One-time budget Intel N100 mini PC: **≈ RM800–1,400** — roughly one year of the subscription, then free thereafter.

Either local-hardware option pays for itself within a single subscription year and needs no recurring MYR payment for "familiar faces"/AI-caption-style features once running the project locally.

## Not found / caveats

- **Tapo Care MY RM pricing**: TP-Link's Malaysia FAQ and pricing pages explicitly defer to the in-app/web-portal pricing widget, which is JS-only and not readable by fetch. Used TP-Link US global USD pricing as the next-best reference; actual MY price may differ.
- **Imou Protect MY RM pricing**: imou.com/my shows no plan prices in static HTML; Shopee Malaysia's official Imou store listing (shopee.com.my/imoumalaysia.os) is JS-only and unreadable. Used the Imou UK/EU store price as reference.
- **Imou Protect and facial recognition**: no documentation found tying Imou Protect to face recognition/familiar faces at all (it appears to be storage + object-class AI detection only); flagged rather than assumed.
- **EZVIZ CloudPlay and faces**: EZVIZ's face-recognition capability is described as a camera-hardware/local feature on select products, not something CloudPlay's subscription tiers unlock; not stated as a CloudPlay-plan feature on the fetched terms/plan pages.
- **N100 mini PC "typical range"**: built from a price-aggregator (BigGo Malaysia) rather than Shopee/Lazada directly, since both marketplaces render prices client-side and returned empty content to the fetch tool.
- **Ring MY availability**: no single authoritative "not available in Malaysia" statement from ring.com itself was found (only its supported-country lists, which omit Malaysia, and general market/press summaries); treated as sufficient evidence of non-availability but not a direct official statement.
- **Arlo/Ring/Google exact 2026-09-12 USD prices**: sourced from a mix of vendor pages and price-tracking sites (subger.com, security.org-style summaries); not all reconfirmed on the vendor's own live page due to 403s (arlo.com blocked WebFetch directly).
