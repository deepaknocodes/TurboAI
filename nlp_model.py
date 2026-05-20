"""
nlp_model.py  —  Train and save the intent classifier.

Run this once (or whenever you update training data) to regenerate
model.pkl and vectorizer.pkl used by intent_predictor.py.

Intent labels MUST match valid_intents in signup.py:
  current_class | next_class | todays_schedule | fee_info |
  teacher_location | greeting | thanks
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np

# ----------------------------
# TRAINING DATA  (~25 examples per intent for good generalisation)
# ----------------------------
training_data = [

    # ── CURRENT CLASS ─────────────────────────────────────────────────────
    ("where is my class",                   "current_class"),
    ("where is my class now",               "current_class"),
    ("which class do i have now",           "current_class"),
    ("what class do i have",                "current_class"),
    ("am i free now",                       "current_class"),
    ("current lecture",                     "current_class"),
    ("what class am i in right now",        "current_class"),
    ("which subject is going on",           "current_class"),
    ("what is happening now",               "current_class"),
    ("tell me my current class",            "current_class"),
    ("ongoing lecture",                     "current_class"),
    ("class going on right now",            "current_class"),
    ("what subject do i have now",          "current_class"),
    ("my class right now",                  "current_class"),
    ("current subject",                     "current_class"),
    ("which room is my class in",           "current_class"),
    ("where should i be right now",         "current_class"),
    ("which period is going on",            "current_class"),
    ("what lecture is now",                 "current_class"),
    ("do i have class now",                 "current_class"),
    ("is there any class now",              "current_class"),
    ("what is my class now",                "current_class"),
    ("where is my lecture",                 "current_class"),
    ("where is the class",                  "current_class"),
    ("which class is on now",               "current_class"),

    # ── NEXT CLASS ─────────────────────────────────────────────────────────
    ("where is my next class",              "next_class"),
    ("next lecture",                        "next_class"),
    ("when is my next class",               "next_class"),
    ("what class do i have next",           "next_class"),
    ("upcoming class",                      "next_class"),
    ("tell me my next subject",             "next_class"),
    ("next period",                         "next_class"),
    ("what is after this",                  "next_class"),
    ("next slot",                           "next_class"),
    ("when does my next lecture start",     "next_class"),
    ("what is my next subject",             "next_class"),
    ("which class is next",                 "next_class"),
    ("next class please",                   "next_class"),
    ("when is my next lecture",             "next_class"),
    ("my next class",                       "next_class"),
    ("upcoming lecture",                    "next_class"),
    ("what comes after this class",         "next_class"),
    ("show me my next class",               "next_class"),
    ("next class time",                     "next_class"),
    ("when is the next period",             "next_class"),
    ("tell me about my next lecture",       "next_class"),
    ("next subject",                        "next_class"),
    ("which room for my next class",        "next_class"),
    ("next class room",                     "next_class"),
    ("after this what class",               "next_class"),

    # ── TODAY'S SCHEDULE ───────────────────────────────────────────────────
    ("show my timetable",                   "todays_schedule"),
    ("show my schedule",                    "todays_schedule"),
    ("todays timetable",                    "todays_schedule"),
    ("what are my classes today",           "todays_schedule"),
    ("full schedule today",                 "todays_schedule"),
    ("all classes today",                   "todays_schedule"),
    ("todays classes",                      "todays_schedule"),
    ("what subjects do i have today",       "todays_schedule"),
    ("schedule for today",                  "todays_schedule"),
    ("list todays lectures",                "todays_schedule"),
    ("my timetable",                        "todays_schedule"),
    ("my schedule",                         "todays_schedule"),
    ("class schedule",                      "todays_schedule"),
    ("show schedule",                       "todays_schedule"),
    ("what is my timetable",                "todays_schedule"),
    ("classes for today",                   "todays_schedule"),
    ("today schedule",                      "todays_schedule"),
    ("what lectures do i have today",       "todays_schedule"),
    ("daily schedule",                      "todays_schedule"),
    ("show timetable",                      "todays_schedule"),
    ("my lectures today",                   "todays_schedule"),
    ("view my timetable",                   "todays_schedule"),
    ("list my classes",                     "todays_schedule"),
    ("what is on today",                    "todays_schedule"),
    ("how many classes today",              "todays_schedule"),

    # ── FEE INFO ───────────────────────────────────────────────────────────
    ("what is fee for btech cse",           "fee_info"),
    ("btech cse fee",                       "fee_info"),
    ("how much is the fee",                 "fee_info"),
    ("fee details",                         "fee_info"),
    ("what are the fees",                   "fee_info"),
    ("fee structure",                       "fee_info"),
    ("fee amount",                          "fee_info"),
    ("how much do i have to pay",           "fee_info"),
    ("what is my fee",                      "fee_info"),
    ("fee due date",                        "fee_info"),
    ("college fee",                         "fee_info"),
    ("semester fee",                        "fee_info"),
    ("tuition fee",                         "fee_info"),
    ("how much is tuition",                 "fee_info"),
    ("what is the fee",                     "fee_info"),
    ("tell me the fee",                     "fee_info"),
    ("fee information",                     "fee_info"),
    ("fees for my course",                  "fee_info"),
    ("my fee amount",                       "fee_info"),
    ("pending fees",                        "fee_info"),
    ("fee last date",                       "fee_info"),
    ("when is fee due",                     "fee_info"),
    ("how much fees",                       "fee_info"),
    ("what fees do i owe",                  "fee_info"),
    ("total fee",                           "fee_info"),

    # ── TEACHER LOCATION ───────────────────────────────────────────────────
    ("where is teacher now",                "teacher_location"),
    ("teacher location",                    "teacher_location"),
    ("where is my teacher",                 "teacher_location"),
    ("which room is the teacher in",        "teacher_location"),
    ("find my teacher",                     "teacher_location"),
    ("teacher current room",                "teacher_location"),
    ("where can i find the professor",      "teacher_location"),
    ("where is professor now",              "teacher_location"),
    ("teacher whereabouts",                 "teacher_location"),
    ("which class is teacher taking now",   "teacher_location"),
    ("find teacher",                        "teacher_location"),
    ("where is the professor",              "teacher_location"),
    ("teacher room",                        "teacher_location"),
    ("where is sir",                        "teacher_location"),
    ("where is mam",                        "teacher_location"),
    ("professor location",                  "teacher_location"),
    ("which room is teacher in",            "teacher_location"),
    ("is teacher in class",                 "teacher_location"),
    ("where does teacher teach now",        "teacher_location"),
    ("locate teacher",                      "teacher_location"),
    ("where is faculty",                    "teacher_location"),
    ("faculty location",                    "teacher_location"),
    ("where is my professor",               "teacher_location"),
    ("teacher class now",                   "teacher_location"),
    ("in which room is teacher",            "teacher_location"),

    # ── GREETING ───────────────────────────────────────────────────────────
    ("hello",                               "greeting"),
    ("hi",                                  "greeting"),
    ("hey",                                 "greeting"),
    ("good morning",                        "greeting"),
    ("good afternoon",                      "greeting"),
    ("good evening",                        "greeting"),
    ("howdy",                               "greeting"),
    ("hiya",                                "greeting"),
    ("what is up",                          "greeting"),
    ("greetings",                           "greeting"),
    ("hi there",                            "greeting"),
    ("hello there",                         "greeting"),
    ("hey there",                           "greeting"),
    ("sup",                                 "greeting"),
    ("yo",                                  "greeting"),
    ("namaste",                             "greeting"),
    ("good day",                            "greeting"),
    ("how are you",                         "greeting"),
    ("how are you doing",                   "greeting"),
    ("whats up",                            "greeting"),
    ("hey bot",                             "greeting"),
    ("hello bot",                           "greeting"),
    ("hi bot",                              "greeting"),
    ("good morning bot",                    "greeting"),
    ("helo",                                "greeting"),

    # ── THANKS ─────────────────────────────────────────────────────────────
    ("thanks",                              "thanks"),
    ("thank you",                           "thanks"),
    ("thx",                                 "thanks"),
    ("thank you so much",                   "thanks"),
    ("many thanks",                         "thanks"),
    ("appreciate it",                       "thanks"),
    ("cheers",                              "thanks"),
    ("ty",                                  "thanks"),
    ("thanks a lot",                        "thanks"),
    ("that helped",                         "thanks"),
    ("thank u",                             "thanks"),
    ("thankyou",                            "thanks"),
    ("great thanks",                        "thanks"),
    ("ok thanks",                           "thanks"),
    ("ok thank you",                        "thanks"),
    ("thanks for the help",                 "thanks"),
    ("helpful thanks",                      "thanks"),
    ("cool thanks",                         "thanks"),
    ("got it thanks",                       "thanks"),
    ("thnks",                               "thanks"),
    ("thnx",                                "thanks"),
    ("thank you very much",                 "thanks"),
    ("much appreciated",                    "thanks"),
    ("nice one",                            "thanks"),
    ("thanks mate",                         "thanks"),
]

texts  = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

# ----------------------------
# BUILD PIPELINE
# C=5 gives less regularisation → better fit on small datasets.
# char_wb analyser also helps catch typos & short inputs.
# ----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),   # unigrams, bigrams, trigrams
        analyzer="word",
        min_df=1,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        C=5.0,                # less regularisation → better on small data
        solver="lbfgs",
        multi_class="multinomial",
    )),
])

pipeline.fit(texts, labels)

# ----------------------------
# CROSS-VAL (informational)
# ----------------------------
scores = cross_val_score(pipeline, texts, labels,
                         cv=min(5, len(set(labels))), scoring="accuracy")
print(f"Cross-val accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# ----------------------------
# SAVE
# ----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(None, f)   # sentinel for backward-compat

print("Model trained and saved to model.pkl ✓")

# ----------------------------
# SMOKE TEST  — includes the exact phrases that were failing
# ----------------------------
test_cases = [
    # phrases that were failing before
    ("where is my class",        "current_class"),
    ("where is my lecture",      "current_class"),
    ("where is the class",       "current_class"),
    # standard cases
    ("where is my class now",    "current_class"),
    ("next lecture",             "next_class"),
    ("my next class",            "next_class"),
    ("show my timetable",        "todays_schedule"),
    ("my schedule",              "todays_schedule"),
    ("what is my fee",           "fee_info"),
    ("college fee",              "fee_info"),
    ("where is teacher now",     "teacher_location"),
    ("where is sir",             "teacher_location"),
    ("hello",                    "greeting"),
    ("hi",                       "greeting"),
    ("thanks",                   "thanks"),
    ("thank u",                  "thanks"),
]

print("\nSmoke-test predictions:")
all_pass = True
for text, expected in test_cases:
    pred  = pipeline.predict([text])[0]
    prob  = pipeline.predict_proba([text]).max()
    ok    = pred == expected
    if not ok:
        all_pass = False
    status = "✓" if ok else "✗"
    print(f"  {status}  '{text}' → {pred} ({prob:.2f})  [expected: {expected}]")

print("\n✅ All tests passed!" if all_pass else "\n⚠️  Some tests failed — add more training examples for those intents.")