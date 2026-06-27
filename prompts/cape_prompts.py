"""
prompts/cape_prompts.py
CAPE 提示集合：原始 Set-A + 扩展 Set-B + 盲集 Set-C
以及四种 uniform prompt 策略
"""

from typing import Dict, List


# ─── 类别描述（用于 uniform prompt 策略）────────────────────────────────────

CLASS_DESCRIPTIONS = {
    # TeacherBehavior
    "guide": "guiding a student individually",
    "answer": "answering a student's question",
    "on-stage interaction": "interacting with students on stage",
    "blackboard-writing": "writing on the blackboard",
    "teacher": "explaining concepts while lecturing",
    "stand": "standing still in the classroom",
    "screen": "presenting slides on a projection screen",
    "blackboard": "referring to content on the blackboard",
    # HandriseReadWrite
    "hand-raise": "raising hand to ask a question",
    "read": "reading a textbook or study material",
    "write": "writing in a notebook",
    # BowTurnHead
    "bow-head": "bowing head down, looking at phone or distracted",
    "turn-head": "turning head sideways, looking away from teacher",
}


# ─── Uniform Prompt 策略 ─────────────────────────────────────────────────────

def get_uniform_prompts(class_name: str, strategy: str) -> List[str]:
    """
    返回给定类别在指定策略下的提示列表
    strategy: "label_only" | "simple" | "action" | "detailed"
    """
    desc = CLASS_DESCRIPTIONS.get(class_name, class_name)
    if strategy == "label_only":
        return [desc]
    elif strategy == "simple":
        return [f"a photo of {desc}"]
    elif strategy == "action":
        return [f"a teacher is {desc}" if class_name in TEACHER_CLASSES
                else f"a student is {desc}"]
    elif strategy == "detailed":
        if class_name in TEACHER_CLASSES:
            return [
                f"a classroom scene where a teacher is {desc}",
                f"a teacher is {desc} during a lecture",
            ]
        else:
            return [
                f"a classroom scene where a student is {desc}",
                f"a student is {desc} during class",
            ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


TEACHER_CLASSES = {
    "guide", "answer", "on-stage interaction",
    "blackboard-writing", "teacher", "stand", "screen", "blackboard"
}

STUDENT_CLASSES = {"hand-raise", "read", "write", "bow-head", "turn-head"}

ALL_STRATEGIES = ["label_only", "simple", "action", "detailed", "cape"]


# ─── CAPE Set-A（论文原始版）────────────────────────────────────────────────

CAPE_SET_A: Dict[str, List[str]] = {
    # ── TeacherBehavior ──────────────────────────────────────────────────
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
    "on-stage interaction": [
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
    "blackboard": [
        "a teacher pointing at the blackboard",
        "a teacher referring to content on the blackboard",
        "a blackboard with writing visible in a classroom",
    ],
    # ── HandriseReadWrite ─────────────────────────────────────────────────
    "hand-raise": [
        "a student raising their hand in a classroom",
        "a student with arm raised to ask a question",
        "a student raising hand to participate in class",
    ],
    "read": [
        "a student reading a textbook at their desk",
        "a student looking down at a book while reading",
        "a student engaged in reading study materials",
    ],
    "write": [
        "a student writing in a notebook",
        "a student taking notes with a pen",
        "a student writing at their desk in class",
    ],
    # ── BowTurnHead ───────────────────────────────────────────────────────
    "bow-head": [
        "a student with their head bowed down",
        "a student looking down at their phone or desk",
        "a student with lowered head not paying attention",
    ],
    "turn-head": [
        "a student turning their head to look sideways",
        "a student looking away from the teacher",
        "a student turning around in their seat",
    ],
}


# ─── CAPE Set-B（独立改写版）────────────────────────────────────────────────

CAPE_SET_B: Dict[str, List[str]] = {
    "guide": [
        "a teacher crouching beside a student to provide individual instruction",
        "an instructor pointing at a student's work to give personalized feedback",
        "a teacher bending over a desk to assist a seated student",
    ],
    "answer": [
        "a teacher facing a student who has raised their hand",
        "an instructor gesturing while verbally responding to a student inquiry",
        "a teacher pausing their lecture to address a student's question",
    ],
    "on-stage interaction": [
        "a teacher calling a student to the front of the room",
        "an instructor standing near the podium while conversing with a student",
        "a teacher and student standing together at the front of the classroom",
    ],
    "blackboard-writing": [
        "a teacher facing the chalkboard with chalk in hand",
        "freshly written text or diagrams visible on a dark chalkboard surface",
        "an instructor's arm extended upward writing on a classroom blackboard",
    ],
    "teacher": [
        "an instructor facing the class with mouth open, mid-sentence",
        "a teacher gesturing with hands while explaining at the front",
        "a person in professional attire addressing a room of students",
    ],
    "stand": [
        "a teacher with arms at their sides, not actively gesturing",
        "an instructor pausing silently between lecture segments",
        "a teacher positioned at the front of the room, stationary",
    ],
    "screen": [
        "a large projected image visible behind the teacher",
        "colorful presentation slides displayed on a wall-mounted screen",
        "a teacher using a remote or pointer near a projection display",
    ],
    "blackboard": [
        "visible chalk writing or diagrams covering a green or black board",
        "a teacher gesturing toward previously written board content",
        "mathematical formulas or text visible on a classroom chalkboard",
    ],
    "hand-raise": [
        "a seated student with one arm extended vertically upward",
        "a child with elbow bent and hand raised above shoulder level",
        "a student reaching their arm skyward to signal the teacher",
    ],
    "read": [
        "a student with eyes directed at an open book on their desk",
        "a child holding a textbook close to their face while reading",
        "a student bent slightly forward, silently reading printed material",
    ],
    "write": [
        "a student gripping a pen or pencil and moving it across paper",
        "a child hunched over a notebook actively forming written characters",
        "a student's hand moving across an exercise book at their desk",
    ],
    "bow-head": [
        "a student with chin nearly touching their chest, gaze directed downward",
        "a person slumped forward with head hanging toward the desk surface",
        "a student whose face is hidden because their head is tilted fully down",
    ],
    "turn-head": [
        "a student with their face oriented toward a classmate rather than the board",
        "a child whose head is rotated more than 45 degrees from the front",
        "a student glancing over their shoulder away from the teacher",
    ],
}


# ─── CAPE Set-C（盲集：仅从类别名称出发，不参考数据）───────────────────────

CAPE_SET_C: Dict[str, List[str]] = {
    "guide": [
        "a person in the role of a guide or mentor",
        "someone guiding or directing another person",
        "a guide assisting someone",
    ],
    "answer": [
        "a person answering or responding",
        "someone providing an answer",
        "a respondent giving information",
    ],
    "on-stage interaction": [
        "people interacting on a stage",
        "on-stage communication between individuals",
        "a person interacting with others on stage",
    ],
    "blackboard-writing": [
        "writing on a blackboard",
        "a person writing on a board",
        "text being written on a dark surface",
    ],
    "teacher": [
        "a person in the role of a teacher or instructor",
        "a teacher standing in front of others",
        "an educator in a teaching role",
    ],
    "stand": [
        "a person standing upright",
        "someone in a standing position",
        "a standing individual",
    ],
    "screen": [
        "a projection screen in use",
        "a screen displaying content",
        "a display or monitor showing information",
    ],
    "blackboard": [
        "a blackboard or chalkboard",
        "a dark writing surface",
        "a board with written content",
    ],
    "hand-raise": [
        "a person raising their hand",
        "someone with their arm raised",
        "a raised hand gesture",
    ],
    "read": [
        "a person reading",
        "someone looking at written text",
        "a reader engaged with a document",
    ],
    "write": [
        "a person writing",
        "someone putting pen to paper",
        "a writing action",
    ],
    "bow-head": [
        "a person with head bowed",
        "someone with their head lowered",
        "a bowed head posture",
    ],
    "turn-head": [
        "a person turning their head",
        "someone looking to the side",
        "a head turned sideways",
    ],
}


# ─── LLM 自动生成提示（GPT-4 / Claude 生成，可在运行时填入）──────────────

CAPE_LLM_GENERATED: Dict[str, List[str]] = {}   # 运行时由 llm_prompt_gen.py 填充


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def get_cape_prompts(class_name: str, set_name: str = "A") -> List[str]:
    mapping = {"A": CAPE_SET_A, "B": CAPE_SET_B, "C": CAPE_SET_C, "LLM": CAPE_LLM_GENERATED}
    assert set_name in mapping, f"Unknown set: {set_name}"
    return mapping[set_name].get(class_name, [CLASS_DESCRIPTIONS.get(class_name, class_name)])


def get_all_class_prompts(
    classes: List[str],
    strategy: str,
    cape_set: str = "A",
) -> Dict[str, List[str]]:
    """
    返回所有类别的提示列表
    strategy: "label_only" | "simple" | "action" | "detailed" | "cape"
    """
    result = {}
    for cls in classes:
        if strategy == "cape":
            result[cls] = get_cape_prompts(cls, cape_set)
        else:
            result[cls] = get_uniform_prompts(cls, strategy)
    return result


if __name__ == "__main__":
    # 快速验证
    from data.scb_dataset import SUBSET_CONFIG
    for subset, cfg in SUBSET_CONFIG.items():
        print(f"\n=== {subset} ===")
        for strategy in ALL_STRATEGIES:
            prompts = get_all_class_prompts(cfg["classes"], strategy)
            total = sum(len(v) for v in prompts.values())
            print(f"  {strategy}: {total} prompts total")
