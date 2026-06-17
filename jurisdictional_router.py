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

def get_clause_template_for_jurisdiction(state_name: str) -> Optional[str]:
    """Return a jurisdiction-specific contract clause template with variable placeholders."""
    profile = classify_jurisdiction(state_name)
    if not profile:
        return None
    
    if profile.category == "prohibitive":
        return f"""
## {profile.state} — PROHIBITIVE ENFORCEMENT (Clause Template)

```
CONFIDENTIALITY AND INVENTION ASSIGNMENT AGREEMENT

This Agreement between [COMPANY NAME] and [EMPLOYEE NAME], effective [START DATE].

WHEREAS, {profile.state} law disfavors restrictive covenants; therefore, the parties agree:

1. **Confidential Information** includes all trade secrets, client lists, pricing, strategic plans, 
   and technical information disclosed during employment.

2. **Invention Assignment:** Employee assigns all inventions, discoveries, and intellectual property 
   developed during employment (or within [6-12] months thereafter, if related to Company business) 
   to Company.

3. **Non-Disclosure:** Employee agrees not to disclose or use confidential information for any purpose 
   other than performing duties for Company.

4. **Return of Materials:** Upon termination, Employee shall return all confidential materials and 
   certify compliance.

[GARDEN LEAVE OPTIONAL: Paid leave of [30-60] days during notice period with full salary continuation.]
```
"""
    
    elif profile.category == "reasonableness":
        return f"""
## {profile.state} — REASONABLENESS ENFORCEMENT (Clause Template)

```
NON-COMPETE AGREEMENT (REASONABLENESS STANDARD)

This Agreement between [COMPANY NAME] and [EMPLOYEE NAME], effective [START DATE].

WHEREAS, [EMPLOYEE NAME] will have access to Company's proprietary information and customer relationships;

NOW, THEREFORE, Employee agrees that for [12-24] months following termination:

1. **Geographic Scope:** Employee shall not engage in any competing business within [X] miles of:
   - Company's principal place of business at [ADDRESS]
   - Any location where Company conducted business during the prior [2] years

2. **Scope of Restriction:** Non-compete applies only to work directly related to:
   - [Specific job functions/business lines]

3. **Legitimate Business Interest:** This restriction protects Company's:
   - Trade secrets and confidential information
   - Customer relationships and goodwill
   - Investment in Employee's training

4. **Confidentiality & IP Assignment:** Employee assigns all inventions developed during employment 
   to Company and agrees not to disclose confidential information.

5. **Consideration:** Employee receives [continued employment | sign-on bonus of $[AMOUNT] | stock options].

[GARDEN LEAVE OPTION: Paid leave of [60-90] days during notice period.]
```
"""
    
    elif profile.category == "statutory":
        threshold = profile.threshold_requirements.get("annual_salary", "N/A") if profile.threshold_requirements else "N/A"
        return f"""
## {profile.state} — STATUTORY ENFORCEMENT (Threshold-Based, Clause Template)

```
NON-COMPETE AGREEMENT (STATUTORY COMPLIANCE)

This Agreement between [COMPANY NAME] and [EMPLOYEE NAME], effective [START DATE].

WHEREAS, Employee's annual compensation is $[SALARY], which meets/exceeds the statutory threshold 
of {threshold};

WHEREAS, Employee is receiving [describe tangible benefit: stock options, sign-on bonus, promotion, 
training program, etc.] as additional consideration;

NOW, THEREFORE, Employee agrees that for [12-24] months following termination:

1. **Non-Compete Restriction:** Employee shall not engage in competing business within [X] miles of 
   Company's principal place of business or any location where Company operates.

2. **Scope:** Restriction applies to [specific job functions and industries].

3. **Statutory Compliance Note:** This agreement complies with {profile.state} statutory requirements 
   and is enforceable only for employees earning above the statutory threshold.

4. **Confidentiality & IP Assignment:** Employee assigns all work-related inventions and agrees not 
   to disclose confidential information.

[GARDEN LEAVE OPTION: Paid leave of [30-60] days during notice period.]
```
"""
    
    return None

def build_multi_jurisdiction_prompt(jurisdictions: List[str]) -> str:
    """Build a system prompt that handles multi-jurisdictional compliance with clause templates."""
    categorized = categorize_all_jurisdictions(jurisdictions)
    
    prompt = """You are a Senior Legal Analyst specializing in multi-state employment law.

## Multi-Jurisdictional Analysis Required

The user has asked about employment contracts spanning MULTIPLE states. Your response MUST:

1. **Identify the enforceability bucket for each state:**
   - PROHIBITIVE: Non-competes are void or nearly unenforceable (e.g., California, Oklahoma)
   - REASONABLENESS: Non-competes are enforceable if they meet "reasonableness" tests (e.g., New York, Texas)
   - STATUTORY: Non-competes require meeting specific thresholds like minimum salary or notice requirements (e.g., Washington, Illinois)

2. **For EACH jurisdiction, provide the corresponding clause template** (with bracketed placeholders like [COMPANY NAME], [X MILES], [SALARY]).
   - Replace placeholders with actual values where possible.
   - NEVER omit templates; include them verbatim so the user has ready-to-adapt legal language.

3. **Provide a COMPARATIVE analysis** showing how the strategy differs by state:
   - Example format: "In CA, prioritize invention assignment and confidentiality; in NY, you may include a narrowly-tailored non-compete lasting 1-2 years."

4. **DO NOT generalize:** If a clause is unenforceable in one state but valid in another, explicitly note the discrepancy.

5. **Recommend a single unified contract template** that is compliant across all relevant jurisdictions (err on the side of the most restrictive state).

---
## Jurisdiction-Specific Profiles & Clause Templates
"""
    
    for j in jurisdictions:
        profile = classify_jurisdiction(j)
        if profile:
            prompt += generate_jurisdiction_summary(profile)
            template = get_clause_template_for_jurisdiction(j)
            if template:
                prompt += template
            prompt += "\n---\n"
    
    return prompt
