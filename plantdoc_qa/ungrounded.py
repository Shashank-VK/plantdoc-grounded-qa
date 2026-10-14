"""Ungrounded baseline agent acting as a negative control for grounding evaluation."""

from dataclasses import dataclass
import re
from typing import List


@dataclass
class UngroundedAnswer:
    """Response produced without retrieval or plant document access."""
    question: str
    answer: str
    citations: List[str]
    status: str = "ungrounded_baseline"


GENERIC_TEMPLATES = [
    (
        r"pump|bfp|flow|suction",
        "Industrial centrifugal pump protocols typically suggest maintaining suction pressure above 1.5 to 2.5 bar and verifying mechanical seal flush cooling prior to energizing. [generic_fluid:std_pump_sop]",
        ["generic_fluid:std_pump_sop"]
    ),
    (
        r"compressor|cc-101|surge|lube oil",
        "Standard multi-stage gas compressor operating guidelines recommend ISO VG 32 or 46 oil with normal bearing temperatures kept under 80°C to 90°C. Anti-surge controllers maintain a typical 10% safety margin. [machinery_best_practice:compressors]",
        ["machinery_best_practice:compressors"]
    ),
    (
        r"cooling tower|ct-3|biocide|water",
        "Evaporative cooling tower maintenance generally involves biocide dosing on an alternating weekly cycle and blowdown conductivity regulation between 2000 and 3000 uS/cm. [general_env:cooling_water]",
        ["general_env:cooling_water"]
    ),
    (
        r"turbine|st-200|steam|resonance",
        "Standard utility steam turbines typically require turning gear operation for several hours prior to roll, with rotor critical speed acceleration kept above 400 RPM/min. [power_standard:turbines]",
        ["power_standard:turbines"]
    ),
    (
        r"conveyor|cv-88|tension|pull cord",
        "Bulk materials conveyor safety standards specify pull cord switch spacing of approximately 25 to 35 meters and periodic counterweight tension verification. [bulk_handling:safety_code]",
        ["bulk_handling:safety_code"]
    ),
    (
        r"distillation|dc-50|tray|column|reboiler",
        "Fractionation columns are normally operated within differential pressure limits of 50 to 100 mbar to prevent liquid entrainment, with reflux ratios adjusted per overhead purity. [chem_eng:fractionation_basics]",
        ["chem_eng:fractionation_basics"]
    ),
    (
        r"effluent|etp|neutralization|ph|cod",
        "Wastewater treatment plants commonly utilize two-stage pH neutralization targeting effluent pH between 6.0 and 9.0 and monitoring COD limits per municipal bylaws. [environmental_rules:wastewater_guide]",
        ["environmental_rules:wastewater_guide"]
    ),
    (
        r"baghouse|dc-12|dust|filter",
        "Pulse-jet baghouse collectors normally operate with clean air differential pressures between 500 and 1500 Pa, using 5 to 7 bar compressed air for on-line pulsing. [filtration:dust_collector_guide]",
        ["filtration:dust_collector_guide"]
    ),
    (
        r"transformer|tr-04|dga|dielectric|oil",
        "Power transformer maintenance guidelines suggest oil breakdown voltage exceeding 40 kV and monitoring dissolved combustible gases like hydrogen and acetylene for thermal faults. [ieee_guide:power_transformers]",
        ["ieee_guide:power_transformers"]
    ),
    (
        r"shift|log|night|handover",
        "Standard plant shift handovers include reviewing open work orders, reporting environmental discharge levels, and logging safety incident counts across units. [operations:shift_handover_std]",
        ["operations:shift_handover_std"]
    ),
]


def answer_ungrounded(question: str) -> UngroundedAnswer:
    """Generate a plausible response using generic engineering templates without retrieval."""
    q_lower = question.lower()

    for pattern, text, citations in GENERIC_TEMPLATES:
        if re.search(pattern, q_lower):
            return UngroundedAnswer(
                question=question,
                answer=text,
                citations=citations,
                status="ungrounded_baseline",
            )

    default_answer = (
        "General industrial facility management recommendations suggest following manufacturer equipment "
        "manuals and site safety compliance protocols. [general_guide:plant_maintenance]"
    )
    return UngroundedAnswer(
        question=question,
        answer=default_answer,
        citations=["general_guide:plant_maintenance"],
        status="ungrounded_baseline",
    )
