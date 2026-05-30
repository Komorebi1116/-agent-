from __future__ import annotations


def route_intent(question: str) -> str:
    lower = question.lower()
    if any(word in lower for word in ["对比", "compare", "comparison", "table", "表格"]):
        return "metric_compare"
    if any(word in lower for word in ["综述", "整理", "快速", "brief", "overview", "summary"]):
        return "model_brief"
    if any(word in lower for word in ["模块", "module", "方法", "哪些", "find"]):
        return "module_search"
    if any(metric.lower() in lower for metric in ["ssim", "psnr", "fid", "lpips", "mae"]):
        return "metric_compare"
    return "qa"

