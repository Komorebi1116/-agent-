Check whether a method card is supported by the cited chunks.

Return strict JSON only:

{
  "evidence_check_passed": true,
  "evidence_issue": ""
}

Check:
- evidence_quote really appears in or is strongly supported by source_chunk_ids.
- evidence_quote supports proposed_solution.
- title is not generic.
- reusable_value_for_user_project is specific.
- the card does not invent unsupported claims.

