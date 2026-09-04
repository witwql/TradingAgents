"""Global Macro Analyst: gold / crude / US yields / US equities factor team.

Produces a structured cross-asset report explaining how the four overnight
factor families transmit into the target's next session, grounded in the
quantitative exposure table from ``get_factor_exposure``. Optional analyst —
wired into the graph only when selected.

Follows the same tool-loop contract as the other analysts: the prompt carries
``MessagesPlaceholder("messages")`` so the model sees its own prior tool
calls/results across LangGraph round-trips, and the node returns the
AIMessage so ``should_continue_macro`` can route to ``tools_macro`` and back
until the model stops calling tools. (Without the placeholder the model
re-calls every tool forever — observed live as 47 identical rounds.)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_availability import (
    analyst_tool_budget,
    tool_rounds_used,
)
from tradingagents.dataflows.global_macro import (
    get_crude_oil_price,
    get_factor_exposure,
    get_gold_price,
    get_money_flow,
    get_us_stock_indices,
    get_us_treasury_yields,
)

_MACRO_TOOLS = [
    get_gold_price,
    get_crude_oil_price,
    get_us_treasury_yields,
    get_us_stock_indices,
    get_money_flow,
    get_factor_exposure,
]

# Keep literal braces out of this text — it goes through ChatPromptTemplate.
_SYSTEM_TEXT = """You are the Global Macro Analyst on an investment research team. Explain how international macro factors influenced, and will likely influence, the target's price action.

Overnight factor families (native tools available — call them, do NOT describe calls in text):
1. get_gold_price — 国际金价 (COMEX GC): risk appetite / hedging demand.
2. get_crude_oil_price — 国际原油 (WTI CL): input costs vs upstream resources.
3. get_us_treasury_yields — 美债 2Y/10Y: discount-rate pressure on growth, yield assets' appeal.
4. get_us_stock_indices — 美股三大指数: leads A-share open sentiment.
5. get_money_flow — 主力资金（超大单+大单）净流入：进攻/出货/吸筹识别，含量价背离警示。
6. get_factor_exposure — 相关系数/β/隔夜综合得分。因子集已含 15 个期货品种
   （外盘 GC/SI/HG/CL/NG + 国内沪金银铜铝锌、螺纹、铁矿、豆粕、焦煤、生猪）与
   主力资金因子，必须调用并引用其数字；表中「联动最强的期货」即该标的的商品锚。

Method: call each factor tool once, call get_factor_exposure for the target,
then write the final report. Noise-level correlations (|r|<0.15) must be
labeled as such instead of being forced into the narrative.

资金面判读（get_money_flow）：主力净占比>2% 且连续净流入+价升 = 趋势健康；
价升而主力流出 = 背离风险，须在报告里显著提示；低位连续净流入+价格滞涨 = 潜在吸筹。
在「隔夜因子概览」表中加一行 主力资金，并把量价背离警示写入「传导分析」。

Final markdown report (only after tools return), titled 「全球宏观因子分析报告」:
一、隔夜因子概览 (表: 因子 | 最新水平 | 日/5日变动 | 方向判断)
二、对标的的传导分析 (引用暴露度数字)
三、明日走势的宏观情景 (基准/乐观/悲观, 含触发条件)
四、宏观综合结论 (一段话 + 偏多/偏空/中性)

{instrument_context}
分析日期: {current_date}（数据均已按此日期防前视过滤）"""


def create_global_macro_analyst(llm):
    llm_with_tools = llm.bind_tools(_MACRO_TOOLS)

    def global_macro_node(state) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_TEXT + "{language_instruction}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(
            instrument_context=get_instrument_context_from_state(state),
            current_date=state.get("trade_date", ""),
            language_instruction=get_language_instruction(),
        )
        # Tool-loop cap enforced at node entry (see market_analyst): past the
        # budget the call goes out WITHOUT bound tools so the model must emit
        # its final report and the router reaches the clear node on its own.
        model = llm_with_tools
        cap = analyst_tool_budget()
        if cap and tool_rounds_used(state["messages_macro"]) >= cap:
            model = llm
        response = (prompt | model).invoke({"messages": state["messages_macro"]})

        report = ""
        if not response.tool_calls:
            report = response.content

        return {"messages_macro": [response], "macro_report": report}

    return global_macro_node
