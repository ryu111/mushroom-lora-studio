# 定義動作和表情列表

# 動作字典
_actions = {
    "standing": "standing still with balanced posture, consistent proportions",    # 穩定站立
    "sitting": "sitting relaxed with legs crossed, consistent proportions",       # 放鬆地盤腿坐下
    "walking": "walking forward with open arms, consistent proportions",          # 向前走路張開雙臂
    "running": "running energetically, consistent proportions",                   # 精力充沛地奔跑
    "jumping": "jumping happily with both arms raised, consistent proportions",   # 開心地跳躍舉起雙臂
    "holding": "holding an object with both hands, consistent proportions"        # 雙手拿著物品
}

# 表情字典
_expressions = {
    "smiling": "gently smiling with curved lips, symmetrical features",                 # 柔和地微笑
    "cheerful": "cheerfully smiling with wide eyes, symmetrical features",              # 快樂地睜大眼睛微笑
    "joyful": "eyes closed with joyful upward curve of mouth, symmetrical features",    # 閉眼並嘴角愉悅上揚
    "playful": "playful grin with one eye winked, symmetrical features",                # 頑皮地眨眼咧嘴笑
    "frowning": "slightly frowning with sad eyes, symmetrical features",                # 眉頭微皺帶有憂傷的眼神
    "neutral": "neutral expression with calm eyes, symmetrical features"                # 平靜的表情
}

import random

# 隨機取得一個動作
def get_random_action_dict() -> dict[str, str]:
    key, value = random.choice(list(_actions.items()))
    return {key: value}

# 隨機取得一個表情
def get_random_expression_dict() -> dict[str, str]:
    key, value = random.choice(list(_expressions.items()))
    return {key: value}

# 預設動作和表情
_default_action_key = "standing"
_default_expression_key = "smiling"

# 獲取預設的動作和表情
def get_default_action_dict() -> dict[str, str]:
    return {_default_action_key: _actions[_default_action_key]}

def get_default_expression_dict() -> dict[str, str]:
    return {_default_expression_key: _expressions[_default_expression_key]}


# 使用詳述說明來生成提示詞
# action_description = _actions[action_key]
# expression_description = _expressions[expression_key]