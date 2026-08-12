# NIK Parser Skill

Use the NIK Parser tools only for structural parsing, local validation, geometric region cropping, and optional extraction from an authorized image.

## Privacy rules

- Never place raw NIK values, OCR text, KTP images, or crop data in logs, artifacts, or ordinary chat replies.
- Return only the masked output provided by the tools unless the user has a specific, legitimate need for sensitive data.
- Do not upload an image or NIK to a third party. `offline` uses local OCR. `vision_model` uses only an explicitly selected enabled Evonic vision model.
- Treat portrait and signature crops only as geometric regions. Never infer identity, ownership, gender, or authenticity from them.

## Limits of the result

A successful result does not establish that a NIK is active, unique, issued by Dukcapil, owned by a person, or that the document is authentic. Confidence and OCR output are review signals only.

## Tool group

This skill is eagerly loaded. Assigning it to an agent automatically assigns these tools: `nik_parse`, `nik_validate`, `nik_detect_crop`, `nik_image_crop`, and `nik_extract_from_image`. Removing the skill removes that group from the agent.
