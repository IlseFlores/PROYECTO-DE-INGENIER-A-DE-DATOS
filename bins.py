import sys


bins = {
    "loan_amnt": [
        {
            "label": "(-inf, 8000)",
            "max": 8000
        },
        {
            "label": "(8000, 12000)",
            "max": 12000
        },
        {
            "label": "(12000, 20000)",
            "max": 20000
        },
        {
            "label": "(20000, inf)",
            "max": sys.maxsize
        },
    ],
    "installment": [
                {
            "label": "(-inf, 260)",
            "max": 260
        },
        {
            "label": "(260, 383)",
            "max": 383
        },
        {
            "label": "(383, 570)",
            "max": 570
        },
        {
            "label": "(570, inf)",
            "max": sys.maxsize
        },
    ],
        "annual_inc": [
                {
            "label": "(-inf, 30)",
            "max": 30
        },
        {
            "label": "(30, 60)",
            "max": 60
        },
        {
            "label": "(60, 120)",
            "max": 120
        },
        {
            "label": "(120, inf)",
            "max": sys.maxsize
        },
    ],
}
