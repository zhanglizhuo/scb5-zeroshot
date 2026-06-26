# prompts.py — Prompt configurations for SCB5 experiments

CLASS_NAMES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]

CLASS_DESCRIPTIONS = {
    "guide":                "guiding or helping students individually",
    "answer":               "answering a student's question",
    "On-stage interaction": "interacting with students in front of the class",
    "blackboard-writing":   "writing on a blackboard with chalk",
    "teacher":              "explaining concepts while lecturing",
    "stand":                "standing still without interacting",
    "screen":               "pointing to or presenting slides on a screen",
    "blackBoard":           "pointing at or referring to the blackboard",
}

# ===== E2: Prompt groups =====
PROMPT_GROUPS = {
    "label_only": ["{cls}"],
    "simple":     ["a photo of {cls}"],
    "action":     ["a teacher is {cls}"],
    "detailed":   [
        "a classroom scene where a teacher is {cls}",
        "a teacher is {cls} during a lecture",
    ],
}

# ===== E4: CAPE (Class-Aware Prompt Ensemble) =====
CAPE_PROMPTS = {
    "guide": [
        "a teacher guiding a student one-on-one",
        "a teacher helping a student at their desk",
        "a teacher walking among students and offering guidance",
    ],
    "answer": [
        "a teacher answering a student's question",
        "a teacher responding to a raised hand in class",
        "a student asking a question and the teacher replying",
    ],
    "On-stage interaction": [
        "a teacher interacting with students at the front of the classroom",
        "a teacher engaging with students on the podium",
        "a teacher and students having a discussion in front of the class",
    ],
    "blackboard-writing": [
        "a teacher writing on a blackboard with chalk",
        "a hand writing equations on a chalkboard",
        "a teacher's back while writing on the blackboard",
    ],
    "teacher": [
        "a teacher standing and explaining a concept",
        "a teacher giving a lecture at the podium",
        "a teacher talking to the class while standing",
    ],
    "stand": [
        "a person standing still in a classroom",
        "a teacher standing at the front without interacting",
        "a teacher standing idle near the podium",
    ],
    "screen": [
        "a teacher pointing at a projection screen",
        "a teacher presenting slides on a screen",
        "a screen displaying a presentation in a classroom",
    ],
    "blackBoard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
    ],
}
