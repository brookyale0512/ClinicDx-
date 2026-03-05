# KB Search Term Rules (Practical)

Use the KB like a retrieval source, not a chatbot.

## Core Rules

- Use **2-4 keyword phrases**, not full sentences.
- Prefer **clinical noun phrases**: `condition intervention context`.
- Include a **specific action term** when possible:
  - `treatment`, `antibiotics`, `dose`, `protocol`, `management`
- Include **common synonym/variant terms** in one query when needed:
  - Example: `postpartum hemorrhage haemorrhage PPH`
- Start with WHO-focused terms; if no hit, retry with alternate disease names for Layer-2 coverage.

## Best-Performing Query Shapes

- `condition + drug/procedure + intent`
  - Example: `PPH oxytocin misoprostol`
- `population + condition + treatment class`
  - Example: `neonatal sepsis antibiotics`
- `condition + protocol + intervention`
  - Example: `snakebite antivenom protocol`
- `condition + specific procedure`
  - Example: `uterine balloon tamponade`

## Terms That Underperformed

- Generic workflow phrases without condition anchor:
  - `pre referral treatment`
  - `child danger signs`
- Very broad or shorthand-only phrases:
  - `MDR TB BPaLM` (can miss depending on tokenization)
- Single unspecific labels for neglected diseases:
  - `noma` alone may fail; use related clinical terms too.

## Fallback Strategy (When No Hit)

1. Add one more concrete clinical token (drug, procedure, or age group).
2. Expand with synonyms in same query:
   - `rabies post exposure prophylaxis`
   - `obstructed labour labor management`
3. Try disease-adjacent terminology:
   - `acute necrotizing gingivitis` (for noma-related retrieval)
4. Keep query short and term-dense; avoid narrative wording.

## Tested Good Examples

- `PPH oxytocin misoprostol`
- `tranexamic acid postpartum`
- `neonatal sepsis antibiotics`
- `eclampsia magnesium sulfate`
- `tuberculosis hepatotoxicity regimen`
- `snakebite antivenom protocol`

## Precision Boost Patterns

- For NTD skin conditions, include disease-specific terms to avoid cross-topic drift:
  - `buruli ulcer rifampicin`
  - `podoconiosis foot hygiene`
  - `tungiasis flea extraction`
  - `noma cancrum oris management`
- For rabies PEP, include at least one rabies anchor:
  - `rabies post exposure vaccine`
  - `rabies immunoglobulin dose`
- Prefer condition anchors over generic terms like `management` alone.

## Quick Template

Use this template when entering search terms:

`[condition] [intervention/drug] [intent/context]`

Examples:
- `severe malaria artesunate treatment`
- `postpartum hemorrhage uterotonic treatment`
- `pneumonia child amoxicillin dose`
