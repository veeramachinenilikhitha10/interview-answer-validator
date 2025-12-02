import os
import json
import re
from ..config import settings

OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY

def local_fallback_score(candidate, passages_texts):
    kws = set()
    for t in passages_texts:
        for w in re.findall(r"\b[a-zA-Z]{4,}\b", t.lower()):
            kws.add(w)

    cand_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", candidate.lower()))
    matches = sorted([k for k in kws if k in cand_words])

    total = max(len(kws), 1)
    score = int(round((len(matches) / total) * 100))

    missing = sorted(list(kws - set(matches)))[:10]
    suggested = "Try mentioning: " + ", ".join(missing[:6]) if missing else "Good coverage."

    evidence = [{"id": f"passage_{i}", "excerpt": t[:240], "score": 0.0} for i, t in enumerate(passages_texts)]

    return {
        "score": score,
        "missing_points": missing[:6],
        "suggested_improvements": suggested,
        "evidence": evidence
    }

async def call_openai_chat(system, user, model="gpt-3.5-turbo", max_tokens=400):
    import openai
    openai.api_key = OPENAI_KEY
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content

async def call_llm_for_evaluation(candidate_answer, question, passages):
    passages_texts = [p["text"] for p in passages]

    if OPENAI_KEY:
        system = (
            "You are an interview coach. Score 0-100, list missing points, and recommend improvements. "
            "Return JSON with: score, missing_points, suggested_improvements, evidence."
        )

        user = (
            f"Question: {question}\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            f"Passages:\n" +
            "\n\n".join([f"[{p['id']}] {p['text'][:500]}" for p in passages]) +
            "\n\nReturn ONLY JSON."
        )

        try:
            raw = await call_openai_chat(system, user)
            return json.loads(raw)
        except:
            return local_fallback_score(candidate_answer, passages_texts)

    return local_fallback_score(candidate_answer, passages_texts)
