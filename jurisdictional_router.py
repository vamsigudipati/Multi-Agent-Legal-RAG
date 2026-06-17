"""
Jurisdictional Routing System for Multi-State Employment Contracts.

Categorizes states into three enforcement buckets and provides jurisdiction-specific
guidance, thresholds, and compliance requirements.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class JurisdictionProfile:
    """Metadata for a jurisdiction."""
    state: str
    category: str  # "prohibitive", "reasonableness", "statutory"
    non_compete_status: str
    key_statutes: List[str]
    enforcement_notes: str
    threshold_requirements: Optional[Dict[str, str]] = None

# ==========================================
# JURISDICTION REGISTRY
# ==========================================

JURISDICTION_REGISTRY = {
    "california": JurisdictionProfile(
        state="California",
        category="prohibitive",
        non_compete_status="Void (with narrow exceptions: sale of business, partnership dissolution)",
        key_statutes=["Cal. Bus. & Prof. Code § 16600"],
        enforcement_notes="California strongly disfavors all employee non-compete agreements. Courts will enforce confidentiality, invention-assignment, and non-solicitation agreements instead.",
        threshold_requirements=None
    ),
    "oklahoma": JurisdictionProfile(
        state="Oklahoma",
        category="prohibitive",
        non_compete_status="Void unless sale of business or dissolution of partnership",
        key_statutes=["Okla. Stat. tit. 15, § 219"],
        enforcement_notes="Oklahoma prohibits non-competes except in narrow circumstances. Focus on trade secret protection and confidentiality.",
        threshold_requirements=None
    ),
    "new york": JurisdictionProfile(
        state="New York",
        category="reasonableness",
        non_compete_status="Enforceable if reasonable in time, area, and line of business",
        key_statutes=["N.Y. Gen. Oblig. Law § 5-322.1"],
        enforcement_notes="New York applies a 'reasonableness' test. Non-competes lasting 6 months to 2 years may be enforceable; anything longer faces scrutiny.",
        threshold_requirements={"duration": "6 months to 2 years recommended", "geographic_scope": "Narrowly tailored to protected interest"}
    ),
    "texas": JurisdictionProfile(
        state="Texas",
        category="reasonableness",
        non_compete_status="Enforceable if reasonable in time, area, and line of business",
        key_statutes=["Tex. Bus. & Com. Code § 15.50"],
        enforcement_notes="Texas enforces reasonable non-competes. Courts examine legitimate business interests and whether the restraint is not greater than necessary.",
        threshold_requirements={"duration": "2 years or less", "geographic_scope": "Must be supported by legitimate business interests"}
    ),
    "washington": JurisdictionProfile(
        state="Washington",
        category="statutory",
        non_compete_status="Enforceable only if minimum salary threshold is met",
        key_statutes=["Wash. Rev. Code § 19.86.140"],
        enforcement_notes="Washington requires high income thresholds. Non-competes for employees earning less than the threshold are void.",
        threshold_requirements={"annual_salary": "$100,000+ (indexed annually)", "notice_requirement": "Written notice at time of employment"}
    ),
    "illinois": JurisdictionProfile(
        state="Illinois",
        category="statutory",
        non_compete_status="Enforceable if employee earns above statutory threshold",
        key_statutes=["740 Ill. Comp. Stat. 140/2"],
        enforcement_notes="Illinois requires the employee to earn at least $75,000 annually. Non-competes for lower-paid employees are void.",
        threshold_requirements={"annual_salary": "$75,000+ (indexed)", "notice_requirement": "Written notice or continued employment benefit"}
    ),
}

def classify_jurisdiction(state_name: str) -> Optional[JurisdictionProfile]:
    """Retrieve the profile for a given state."""
    normalized = state_name.lower().strip()
    return JURISDICTION_REGISTRY.get(normalized)

def categorize_all_jurisdictions(jurisdictions: List[str]) -> Dict[str, List[str]]:
    """Categorize multiple jurisdictions into buckets."""
    categorized = {"prohibitive": [], "reasonableness": [], "statutory": []}
    
    for j in jurisdictions:
        profile = classify_jurisdiction(j)
        if profile:
            categorized[profile.category].append(profile.state)
        else:
            # If unknown, treat as reasonableness (mid-tier caution)
            categorized["reasonableness"].append(j)
    
    return categorized

def generate_jurisdiction_summary(profile: JurisdictionProfile) -> str:
    """Generate a brief compliance summary for a jurisdiction."""
    summary = f"""
**{profile.state}** ({profile.category.title()})
- Non-Compete Status: {profile.non_compete_status}
- Key Statute(s): {', '.join(profile.key_statutes)}
- Enforcement Notes: {profile.enforcement_notes}
"""
    if profile.threshold_requirements:
        summary += "- Threshold Requirements:\n"
        for req, value in profile.threshold_requirements.items():
            summary += f"  - {req.replace('_', ' ').title()}: {value}\n"
    
    return summary

def build_multi_jurisdiction_prompt(jurisdictions: List[str]) -> str:
    """Build a system prompt that handles multi-jurisdictional compliance."""
    categorized = categorize_all_jurisdictions(jurisdictions)
    
    prompt = """You are a Senior Legal Analyst specializing in multi-state employment law.

## Multi-Jurisdictional Analysis Required

The user has asked about employment contracts spanning MULTIPLE states. Your response MUST:

1. **Identify the enforceability bucket for each state:**
   - PROHIBITIVE: Non-competes are void or nearly unenforceable (e.g., California, Oklahoma)
   - REASONABLENESS: Non-competes are enforceable if they meet "reasonableness" tests (e.g., New York, Texas)
   - STATUTORY: Non-competes require meeting specific thresholds like minimum salary or notice requirements (e.g., Washington, Illinois)

2. **Provide a COMPARATIVE table or bulleted list** showing how the strategy differs by state:
   - Example format: "In CA, prioritize invention assignment and confidentiality; in NY, you may include a narrowly-tailored non-compete lasting 1-2 years."

3. **DO NOT generalize:** If a clause is unenforceable in one state but valid in another, explicitly note the discrepancy.

4. **Recommend a single unified contract template** that is compliant across all relevant jurisdictions (err on the side of the most restrictive state).

---
## Jurisdiction-Specific Guidance
"""
    
    for j in jurisdictions:
        profile = classify_jurisdiction(j)
        if profile:
            prompt += generate_jurisdiction_summary(profile)
    
    return prompt
