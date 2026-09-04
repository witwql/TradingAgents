from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.agents.utils.tool_availability import (
    analyst_tool_budget,
    available_tools,
    tool_rounds_used,
)


def create_news_analyst(llm):
    # Bound once per run; the filtered list never advertises a tool whose
    # vendor is unconfigured (see tool_availability) so bind time stays
    # deterministic instead of failing per call.
    tools = available_tools(
        [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]
    )
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    bound_names = {tool.name for tool in tools}

    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        # Prompt segments stay in lockstep with the bound tools: an unbound
        # tool must never be advertised, or the LLM calls a tool that will
        # never execute.
        tool_docs = {
            "get_news": f"get_news(ticker, start_date, end_date) for {asset_label}-specific news by ticker symbol",
            "get_global_news": "get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news",
            "get_macro_indicators": "get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve')",
            "get_prediction_markets": "get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events)",
        }
        tool_guide = ""
        if bound_names:
            tool_guide = "Use the available tools: " + ", ".join(
                tool_docs[name] for name in tool_docs if name in bound_names
            ) + ". "

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. "
            + tool_guide
            + "Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Tool-loop cap enforced at node entry (see market_analyst): past the
        # budget the call goes out WITHOUT bound tools so the model must emit
        # its final report and the router reaches the clear node on its own.
        cap = analyst_tool_budget()
        model = llm_with_tools
        if cap and tool_rounds_used(state["messages_news"]) >= cap:
            model = llm

        chain = prompt | model
        result = chain.invoke(state["messages_news"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages_news": [result],
            "news_report": report,
        }

    return news_analyst_node
