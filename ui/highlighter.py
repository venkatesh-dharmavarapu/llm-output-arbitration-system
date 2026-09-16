import html
from typing import List, Union, Dict, Any

SEVERITY_COLORS = {
    "critical": "#ff4d4f",
    "high": "#ff7a45",
    "medium": "#ffa940",
    "low": "#ffec3d"
}

def render_annotated_text(original_text: str, confirmed_issues: List[Union[Dict[str, Any], Any]]) -> str:
    """Wraps text quotes flagged by the adjudicator with styled HTML highlight spans."""
    escaped_text = html.escape(original_text)
    
    # Helper to extract fields whether item is dict or Pydantic object
    def get_val(item, key, default=""):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    # Sort issues by quote length descending to prevent partial substring collision
    sorted_issues = sorted(confirmed_issues, key=lambda x: len(str(get_val(x, "quote", ""))), reverse=True)
    
    for issue in sorted_issues:
        raw_quote = str(get_val(issue, "quote", "")).strip()
        if not raw_quote:
            continue
            
        escaped_quote = html.escape(raw_quote)
        if escaped_quote in escaped_text:
            sev = str(get_val(issue, "severity", "medium")).lower()
            color = SEVERITY_COLORS.get(sev, "#ffa940")
            dim = str(get_val(issue, "dimension", "issue")).upper()
            reasoning = html.escape(str(get_val(issue, "evidence_reasoning", "")))
            
            badge = (
                f"<span style='background-color: {color}33; border-bottom: 2px solid {color}; "
                f"padding: 2px 4px; border-radius: 4px;' title='[{dim}] {reasoning}'>"
                f"{escaped_quote}</span>"
            )
            escaped_text = escaped_text.replace(escaped_quote, badge, 1)
            
    return f"<div style='font-size: 1.05rem; line-height: 1.7; padding: 12px; background: rgba(128,128,128,0.08); border-radius: 8px;'>{escaped_text}</div>"