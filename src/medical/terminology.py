"""
Centralized German medical terminology database.

Single source of truth for all medical term lookups used by ASR evaluation,
summarization quality scoring, and error analysis.
"""

MEDICAL_TERMS: dict[str, list[str]] = {
    # ── Symptoms ──────────────────────────────────────────────────────────────
    "symptoms": [
        "schmerz", "schmerzen", "kopfschmerzen", "bauchschmerzen",
        "rückenschmerzen", "halsschmerzen", "brustschmerzen", "gelenkschmerzen",
        "fieber", "husten", "schnupfen", "übelkeit", "erbrechen", "brechreiz",
        "durchfall", "verstopfung", "schwindel", "schwindelgefühl", "müdigkeit",
        "erschöpfung", "appetitlosigkeit", "gewichtsverlust", "atemnot",
        "kurzatmigkeit", "herzrasen", "herzstolpern", "blutungen", "blutung",
        "schwellung", "ödeme", "kribbeln", "taubheitsgefühl", "zittern",
        "krampf", "krämpfe", "juckreiz", "ausschlag", "hautausschlag",
        "rötung", "entzündung", "schlaflosigkeit", "benommenheit",
    ],

    # ── Diagnosis / Conditions ─────────────────────────────────────────────
    "diagnosis": [
        "diagnose", "befund", "befunde", "diagnosen", "erkrankung",
        "krankheit", "erkrankungen", "syndrom", "störung", "leiden",
        "infektion", "bakteriell", "viral", "entzündung", "entzündlich",
        "chronisch", "akut", "gutartig", "bösartig",
        "krebs", "tumor", "karzinom", "malignom", "metastase",
        "diabetes", "diabetisch", "insulinresistenz",
        "hypertonie", "hypotonie", "bluthochdruck", "herzinsuffizienz",
        "herzinfarkt", "herzerkrankung", "herzfehler",
        "schlaganfall", "apoplex", "thrombose", "embolie",
        "arthrose", "arthritis", "rheuma", "rheumatoid",
        "asthma", "bronchitis", "pneumonie", "lungenentzündung",
        "gastritis", "magengeschwür", "colitis", "morbus crohn",
        "niereninsuffizienz", "leberinsuffizienz", "zirrhose",
        "depression", "angststörung", "psychose", "demenz", "alzheimer",
        "epilepsie", "migräne", "parkinson",
        "hypothyreose", "hyperthyreose", "schilddrüsenerkrankung",
        "osteoporose", "fraktur", "knochenbruch",
        "anämie", "blutarmut",
        "allergie", "allergisch", "unverträglichkeit",
        "übergewicht", "adipositas",
    ],

    # ── Medications ────────────────────────────────────────────────────────
    "medication": [
        "medikament", "medikamente", "medikation", "arzneimittel",
        "tablette", "tabletten", "kapsel", "kapseln", "tropfen",
        "spritze", "injektion", "infusion", "pflaster",
        "antibiotikum", "antibiotika",
        "schmerzmittel", "analgetikum", "analgetika",
        "ibuprofen", "paracetamol", "aspirin",
        "antihypertensivum", "betablocker", "ace-hemmer",
        "insulin", "metformin",
        "kortison", "kortikosteroid", "steroid",
        "antidepressivum", "antidepressiva",
        "beruhigungsmittel", "schlafmittel",
        "diuretikum", "diuretika", "entwässerungsmittel",
        "impfstoff", "impfung",
        "dosis", "dosierung", "einnahme",
        "wirkstoff", "wirkung", "nebenwirkung", "nebenwirkungen",
        "kontraindikation", "wechselwirkung", "allergiereaktionen",
        "rezept", "verschreibung", "verordnung",
    ],

    # ── Treatment / Procedures ─────────────────────────────────────────────
    "treatment": [
        "therapie", "therapien", "behandlung", "behandlungen",
        "operation", "eingriff", "chirurgie", "chirurgisch",
        "physiotherapie", "krankengymnastik", "rehabilitation", "reha",
        "chemotherapie", "strahlentherapie", "bestrahlung",
        "dialyse", "transfusion",
        "untersuchung", "untersuchungen", "kontrolle",
        "überweisung", "konsultation", "nachsorge",
        "prävention", "vorbeugung", "impfschutz",
        "erste hilfe", "notfallbehandlung",
        "wundversorgung", "verbandswechsel",
        "massage", "akupunktur",
    ],

    # ── Diagnostics / Tests ────────────────────────────────────────────────
    "diagnostics": [
        "blutuntersuchung", "blutabnahme", "blutbild", "bluttest",
        "röntgen", "röntgenaufnahme",
        "mrt", "magnetresonanztomographie", "tomographie",
        "ct", "computertomographie",
        "ultraschall", "sonographie", "echokardiographie",
        "ekg", "elektrokardiogramm",
        "gastroskopie", "koloskopie", "endoskopie",
        "biopsie", "histologie",
        "urinuntersuchung", "urinprobe",
        "stuhluntersuchung",
        "lungenfunktionstest", "spirometrie",
        "laborwerte", "laborergebnis",
    ],

    # ── Body Parts ─────────────────────────────────────────────────────────
    "body_parts": [
        "kopf", "gehirn", "hirn",
        "hals", "nacken", "kehle",
        "brust", "brustkorb", "thorax",
        "bauch", "abdomen",
        "rücken", "wirbelsäule", "lendenwirbel",
        "herz", "herzmuskel",
        "lunge", "lungen", "bronchien",
        "magen", "darm", "dickdarm", "dünndarm",
        "leber", "gallenblase", "galle",
        "niere", "nieren", "blase",
        "milz", "pankreas", "bauchspeicheldrüse",
        "schilddrüse",
        "arm", "arme", "schulter", "ellenbogen", "handgelenk", "hand",
        "bein", "beine", "hüfte", "knie", "knöchel", "fuß", "füße",
        "auge", "augen", "ohr", "ohren", "nase", "mund", "zähne", "zahn",
        "haut", "muskeln", "knochen", "gelenke", "sehnen",
        "blutgefäße", "arterien", "venen", "nerven",
    ],

    # ── Vitals ─────────────────────────────────────────────────────────────
    "vitals": [
        "blutdruck", "systolisch", "diastolisch",
        "puls", "herzfrequenz", "herzrate",
        "temperatur", "körpertemperatur", "fieberkurve",
        "atemfrequenz", "atemrate",
        "sauerstoffsättigung", "sauerstoff", "o2-sättigung",
        "blutzucker", "glukose", "hba1c",
        "gewicht", "körpergewicht", "bmi",
        "körpergröße",
    ],

    # ── Clinical Setting ───────────────────────────────────────────────────
    "clinical": [
        "arzt", "ärztin", "doktor",
        "patient", "patientin", "kranke",
        "krankenhaus", "klinik", "hospital",
        "notaufnahme", "notfall",
        "ambulanz", "ambulant", "stationär",
        "intensivstation", "icu",
        "pflegepersonal", "krankenschwester", "pfleger",
        "facharzt", "spezialist", "hausarzt",
        "praxis", "arztpraxis",
        "anamnese", "vorgeschichte", "krankengeschichte",
        "diagnosestellung", "zweitmeinung",
    ],

    # ── Critical / Safety Terms ────────────────────────────────────────────
    "critical": [
        "nicht", "kein", "keine",
        "allergie", "allergisch",
        "kontraindiziert", "kontraindikation",
        "unverträglich", "unverträglichkeit",
        "überdosierung", "vergiftung",
        "notfall", "lebensgefährlich", "lebensbedrohlich",
        "bewusstlos", "reanimation",
    ],
}


def get_all_terms() -> set[str]:
    """Return a flat set of every medical term across all categories."""
    return {term for terms in MEDICAL_TERMS.values() for term in terms}


def extract_medical_terms(text: str) -> set[str]:
    """
    Extract German medical terms found in *text*.

    Uses substring matching to handle compound words (common in German).
    """
    text_lower = text.lower()
    all_terms = get_all_terms()
    found: set[str] = set()
    for term in all_terms:
        if term in text_lower:
            found.add(term)
    return found


def medical_term_preservation(reference: str, hypothesis: str) -> float:
    """
    Fraction of medical terms present in *reference* that also appear in *hypothesis*.

    Returns 1.0 when reference contains no medical terms (nothing to preserve).
    """
    ref_terms = extract_medical_terms(reference)
    if not ref_terms:
        return 1.0
    hyp_terms = extract_medical_terms(hypothesis)
    return len(ref_terms & hyp_terms) / len(ref_terms)
