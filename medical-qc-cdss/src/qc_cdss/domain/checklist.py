"""LLM 内涵质控清单。

规则引擎覆盖"可判定"问题，清单覆盖"需语义理解"问题。每条清单项都是一个
独立的模型调用单元（可被缓存、可并行、可单独评测），这样：
- 单条失败不影响其它项（局部降级）；
- 可以针对单项做 prompt 迭代与 A/B；
- 评测集能按检查项拆分，定位是"哪一类问题"退化了。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChecklistItem", "DEFAULT_CHECKLIST", "get_checklist"]


@dataclass(frozen=True)
class ChecklistItem:
    code: str
    title: str
    category: str
    severity: str
    sections: tuple[str, ...]
    instruction: str
    require_evidence: bool = True


DEFAULT_CHECKLIST: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        code="QC-LLM-01",
        title="主诉与现病史一致性",
        category="consistency",
        severity="high",
        sections=("主诉", "现病史"),
        instruction=(
            "判断主诉所描述的症状、部位、性质、持续时间，是否与现病史的叙述一致。"
            "重点检查：主诉症状在现病史中是否被否定、时间是否矛盾、部位是否漂移。"
        ),
    ),
    ChecklistItem(
        code="QC-LLM-02",
        title="诊断依据充分性",
        category="logic",
        severity="high",
        sections=("现病史", "既往史", "辅助检查", "初步诊断"),
        instruction=(
            "判断初步诊断是否有充分依据（症状、体征、实验室/影像检查）。"
            "若诊断缺乏关键检查结果支持，或诊断与检查结论矛盾，应指出。"
        ),
    ),
    ChecklistItem(
        code="QC-LLM-03",
        title="查体与诊断相符性",
        category="logic",
        severity="medium",
        sections=("体格检查", "专科检查", "初步诊断"),
        instruction=(
            "判断体格检查/专科检查的阳性体征是否支持诊断，是否存在与诊断明显不符"
            "或应查未查的关键体征（如肺炎未记录肺部啰音）。"
        ),
    ),
    ChecklistItem(
        code="QC-LLM-04",
        title="用药合理性与禁忌",
        category="safety",
        severity="high",
        sections=("医嘱", "既往史", "初步诊断"),
        instruction=(
            "结合诊断与既往史（过敏史、肝肾功能），判断用药是否存在明确禁忌、"
            "重复用药或剂量异常。仅在依据充分时提出，避免臆测。"
        ),
    ),
    ChecklistItem(
        code="QC-LLM-05",
        title="危急重症风险提示",
        category="safety",
        severity="high",
        sections=("主诉", "现病史", "体格检查", "辅助检查"),
        instruction=(
            "识别病历中提示危急重症的风险信号（如胸痛伴大汗、意识改变、"
            "血压/血糖/血钾危急值），判断病历是否记录了相应处置或风险告知。"
        ),
    ),
    ChecklistItem(
        code="QC-LLM-06",
        title="诊疗计划完整性",
        category="completeness",
        severity="medium",
        sections=("诊疗计划", "初步诊断"),
        instruction=(
            "判断诊疗计划是否包含检查计划、治疗方案、病情告知与随访安排，"
            "是否与诊断对应。"
        ),
    ),
)


def get_checklist(version: str | None = None) -> tuple[ChecklistItem, ...]:
    """按版本返回清单。当前仅 v1，保留扩展点用于灰度。"""
    _ = version
    return DEFAULT_CHECKLIST
