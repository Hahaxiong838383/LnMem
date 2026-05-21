"""Chinese parser benchmark sentences. 10 句 P0 槽位样本。"""
from __future__ import annotations

SENTENCES = [
    {
        "text": "昨晚11点我和小李在公司吃饭。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "吃饭",
            "when_time": "昨晚11点",
            "where_location": "公司",
        },
    },
    {
        "text": "明天上午10点我和王总在会议室开评审会。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "开评审会",
            "when_time": "明天上午10点",
            "where_location": "会议室",
        },
    },
    {
        "text": "上周日我在杭州西湖跑步。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "跑步",
            "when_time": "上周日",
            "where_location": "杭州西湖",
        },
    },
    {
        "text": "今天中午陈川在家写周报。",
        "expected_p0": {
            "who_actor": "陈川",
            "what_action": "写周报",
            "when_time": "今天中午",
            "where_location": "家",
        },
    },
    {
        "text": "前天晚上我和妈妈在北京站见面。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "见面",
            "when_time": "前天晚上",
            "where_location": "北京站",
        },
    },
    {
        "text": "刚才我在微信上和小王聊项目进度。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "聊项目进度",
            "when_time": "刚才",
            "where_location": "微信上",
        },
    },
    {
        "text": "六月一号我在上海办公室提交合同。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "提交合同",
            "when_time": "六月一号",
            "where_location": "上海办公室",
        },
    },
    {
        "text": "明晚8点我和产品团队在腾讯会议复盘demo。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "复盘demo",
            "when_time": "明晚8点",
            "where_location": "腾讯会议",
        },
    },
    {
        "text": "去年国庆我在成都看熊猫。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "看熊猫",
            "when_time": "去年国庆",
            "where_location": "成都",
        },
    },
    {
        "text": "今晚我在书房整理发票。",
        "expected_p0": {
            "who_actor": "我",
            "what_action": "整理发票",
            "when_time": "今晚",
            "where_location": "书房",
        },
    },
]
