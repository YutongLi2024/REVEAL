DEFAULT_PROMPT_TEMPLATE = (
    "An item image of a {category}. "
    "Item name: {title}. "
    "Users often mention: {keywords}. "
    "Focus on the overall appearance of the item."
)

DEFAULT_EVAL_INSTR = (
    "Assess whether current prompt effectively extracts visual features for correct recommendation "
    "by comparing prediction and ground truth results. "
    "If recommendation errors occur, analyze whether the prompt misses preference relevant visual attributes "
    "and suggest improvements for prompt."
)

