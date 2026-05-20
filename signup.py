"""
signup.py  —  Main Flask application for the college chatbot portal.

Changes from original:
  • _build_chatbot_response: added teacher_location handler (was missing),
    fixed intent name "todays_schedule" (was "timetable" in old model).
  • valid_intents whitelist updated to include "teacher_location".
  • Rule-based layer expanded so "greeting" / "thanks" are also recognised
    from the ML model path (not just hardcoded strings).
  • Minor: used db.session.get() instead of deprecated Query.get() on
    SQLAlchemy 2.x to silence deprecation warnings.
"""

from flask import Flask, render_template, request, redirect, session, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from intent_predictor import predict

app = Flask(__name__)
app.secret_key = "Deepak_secure_key_2026!@#"   # Move to env / .env in production
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///signup.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ==================================================
# MODELS
# ==================================================

class User(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(100), unique=True, nullable=False)
    password    = db.Column(db.String(200), nullable=False)
    role        = db.Column(db.String(20), default='user')
    section_id  = db.Column(db.Integer, db.ForeignKey('section.id'), nullable=True)

    chats       = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')
    section     = db.relationship('Section', backref='students')


class Chat(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    message     = db.Column(db.String(500), nullable=False)
    response    = db.Column(db.String(500), nullable=False)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class Course(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    name    = db.Column(db.String(100), unique=True, nullable=False)

    years   = db.relationship('Year', backref='course', lazy=True)
    fees    = db.relationship('FeeStructure', backref='course', lazy=True)


class Teacher(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    timetables  = db.relationship('TeacherTimetable', backref='teacher', lazy=True)


class TeacherTimetable(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    subject     = db.Column(db.String(100), nullable=False)
    room        = db.Column(db.String(20))
    day         = db.Column(db.String(20), nullable=False)
    start_time  = db.Column(db.String(20), nullable=False)
    end_time    = db.Column(db.String(20), nullable=False)
    teacher_id  = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)


class Year(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(50), nullable=False)
    course_id   = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)

    sections    = db.relationship('Section', backref='year', lazy=True)
    fees        = db.relationship('FeeStructure', backref='year', lazy=True)


class Section(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(20), nullable=False)
    year_id     = db.Column(db.Integer, db.ForeignKey('year.id'), nullable=False)

    timetables  = db.relationship('Timetable', backref='section', lazy=True)


class Timetable(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    subject     = db.Column(db.String(100), nullable=False)
    teacher     = db.Column(db.String(100))
    start_time  = db.Column(db.String(20), nullable=False)
    end_time    = db.Column(db.String(20), nullable=False)
    room        = db.Column(db.String(20))
    day         = db.Column(db.String(20), nullable=False)
    section_id  = db.Column(db.Integer, db.ForeignKey('section.id'), nullable=False)


class FeeStructure(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    amount      = db.Column(db.Integer, nullable=False)
    last_date   = db.Column(db.String(50))
    course_id   = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    year_id     = db.Column(db.Integer, db.ForeignKey('year.id'), nullable=False)


# ==================================================
# AUTH DECORATORS
# ==================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return render_template('403.html'), 403
        return f(*args, **kwargs)
    return decorated


# ==================================================
# CHATBOT HELPERS
# ==================================================

def _time_to_minutes(t: str) -> int:
    """Convert 'HH:MM' string to total minutes for easy comparison."""
    try:
        h, m = map(int, t.strip().split(':'))
        return h * 60 + m
    except Exception:
        return -1


def _build_chatbot_response(
    intent: str,
    timetables,
    current_day: str,
    current_time: str,
) -> str:
    """Return the chatbot reply string for a given intent."""

    now_minutes   = _time_to_minutes(current_time)
    today_classes = [t for t in timetables if t.day.lower() == current_day.lower()]

    # ── current_class ────────────────────────────────────────────────────
    if intent == 'current_class':
        current = [
            t for t in today_classes
            if _time_to_minutes(t.start_time) <= now_minutes <= _time_to_minutes(t.end_time)
        ]
        if current:
            c = current[0]
            return (
                f"📚 You are currently in **{c.subject}** taught by {c.teacher}. "
                f"Room: {c.room} | {c.start_time} – {c.end_time}"
            )
        return "You have no class right now. Enjoy your break! 🎉"

    # ── next_class ───────────────────────────────────────────────────────
    elif intent == 'next_class':
        upcoming = sorted(
            [t for t in today_classes if _time_to_minutes(t.start_time) > now_minutes],
            key=lambda t: _time_to_minutes(t.start_time),
        )
        if upcoming:
            n = upcoming[0]
            return (
                f"⏭ Your next class is **{n.subject}** with {n.teacher} "
                f"at {n.start_time} in Room {n.room}."
            )
        return "No more classes for today! 🏁"

    # ── todays_schedule ──────────────────────────────────────────────────
    elif intent == 'todays_schedule':
        if not today_classes:
            return f"No classes scheduled for {current_day}."
        sorted_classes = sorted(today_classes, key=lambda t: _time_to_minutes(t.start_time))
        lines = [f"📅 **{current_day}'s Schedule:**"]
        for t in sorted_classes:
            lines.append(
                f"• {t.start_time}–{t.end_time}: {t.subject} ({t.teacher}) – Room {t.room}"
            )
        return "\n".join(lines)

    # ── fee_info ─────────────────────────────────────────────────────────
    elif intent == 'fee_info':
        user    = db.session.get(User, session['user_id'])
        section = db.session.get(Section, user.section_id) if user else None
        if section:
            year = db.session.get(Year, section.year_id)
            fee  = (
                FeeStructure.query.filter_by(
                    course_id=year.course_id,
                    year_id=year.id,
                ).first()
                if year else None
            )
            if fee:
                due = f" (Due: {fee.last_date})" if fee.last_date else ""
                return f"💰 Your fee is ₹{fee.amount}{due}."
        return "Fee information not found. Please contact the admin."

    # ── teacher_location ─────────────────────────────────────────────────
    elif intent == 'teacher_location':
        now_minutes = _time_to_minutes(current_time)
        # Find all teacher timetable entries for right now
        all_teacher_slots = TeacherTimetable.query.filter_by(day=current_day).all()
        active = [
            s for s in all_teacher_slots
            if _time_to_minutes(s.start_time) <= now_minutes <= _time_to_minutes(s.end_time)
        ]
        if active:
            lines = ["📍 **Teachers currently in class:**"]
            for s in active:
                lines.append(
                    f"• {s.teacher.name} — {s.subject} in Room {s.room} "
                    f"({s.start_time}–{s.end_time})"
                )
            return "\n".join(lines)
        return "No teachers appear to be in class right now."

    # ── greeting ─────────────────────────────────────────────────────────
    elif intent == 'greeting':
        return "👋 Hello! I can help you with your schedule, classes, and fees. What would you like to know?"

    # ── thanks ───────────────────────────────────────────────────────────
    elif intent == 'thanks':
        return "You're welcome! 😊 Let me know if you need anything else."

    # ── fallback ─────────────────────────────────────────────────────────
    return "Sorry, I didn't understand that. Try asking about your schedule, fees, or current class."


# ==================================================
# HOME
# ==================================================

@app.route('/')
def index():
    return render_template('index.html')


# ==================================================
# CREATE ADMIN  (one-time setup — protect or remove in production)
# ==================================================

@app.route('/create_admin')
def create_admin():
    if User.query.filter_by(username="deepak@admin").first():
        return "Admin already exists.", 400
    admin = User(
        username=generate_password_hash("deepak@admin"),
        password=generate_password_hash("admin123"),
        role="admin",
    )
    db.session.add(admin)
    db.session.commit()
    return "Admin Created"


# ==================================================
# SIGNUP
# ==================================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username   = request.form.get('username', '').strip()
        password   = request.form.get('password', '').strip()
        section_id = request.form.get('section_id', '').strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for('signup'))
        if not section_id:
            flash("Please select a section.", "error")
            return redirect(url_for('signup'))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return redirect(url_for('signup'))

        db.session.add(User(
            username   = username,
            password   = generate_password_hash(password),
            section_id = section_id,
        ))
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))

    courses = Course.query.all()
    return render_template('signup.html', courses=courses)


# ==================================================
# LOGIN
# ==================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id']  = user.id
            session['username'] = user.username
            session['role']     = user.role
            return redirect(url_for('admin') if user.role == 'admin' else url_for('chatbot'))

        flash("Invalid username or password.", "error")
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/get-years/<int:course_id>')
def get_years(course_id):
    years = Year.query.filter_by(course_id=course_id).all()
    return {"years": [{"id": y.id, "name": y.name} for y in years]}


@app.route('/get-sections/<int:year_id>')
def get_sections(year_id):
    sections = Section.query.filter_by(year_id=year_id).all()
    return {"sections": [{"id": s.id, "name": s.name} for s in sections]}


# ==================================================
# CHATBOT
# ==================================================

_VALID_INTENTS = {
    "current_class",
    "next_class",
    "todays_schedule",
    "fee_info",
    "teacher_location",
    "greeting",
    "thanks",
}

_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening"
}

_THANKS_PHRASES = {
    "thanks",
    "thank you",
    "thx",
    "ty"
}


@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():

    if request.method == 'POST':

        # ==================================================
        # GET USER MESSAGE
        # ==================================================
        message = request.form.get('message', '').strip()

        if not message:
            flash("Please enter a message.", "warning")
            return redirect(url_for('chatbot'))

        message_lc = message.lower()

        # ==================================================
        # USER DATA
        # ==================================================
        user = db.session.get(User, session['user_id'])

        timetables = Timetable.query.filter_by(
            section_id=user.section_id
        ).all()

        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")

        # ==================================================
        # TEACHER NAME EXTRACTION
        # ==================================================
        session["teacher_name"] = None

        teachers = Teacher.query.all()

        for t in teachers:

            teacher_name = t.name.lower()

            # exact match
            if teacher_name in message_lc:
                session["teacher_name"] = t.name
                break

            # partial match
            teacher_words = teacher_name.split()

            if any(word in message_lc.split() for word in teacher_words):
                session["teacher_name"] = t.name
                break

        # ==================================================
        # RULE BASED GREETING
        # ==================================================
        if message_lc in _GREETING_PHRASES:

            response = "👋 Hello! How can I help you today?"

        elif message_lc in _THANKS_PHRASES:

            response = "😊 You're welcome!"

        else:

            # ==================================================
            # INTENT PREDICTION
            # ==================================================
            result = predict(
                message_lc,
                confidence_threshold=0.55
            )

            intent = result.intent
            confidence = result.confidence
            is_confident = result.is_confident

            print(
                f"[Chatbot] '{message}' → intent={intent} confidence={confidence:.2f}"
            )

            response = "Processing your request..."  # fallback
            # ==================================================
            # MANUAL RULES FOR COMMON QUESTIONS
            # ==================================================

            if (
                "where is" in message_lc
                or "where's" in message_lc
                or "location of" in message_lc
            ):

                intent = "teacher_location"

                is_confident = True

            elif (
                "current class" in message_lc
                or "class now" in message_lc
                or "my class now" in message_lc
            ):

                intent = "current_class"

                is_confident = True

            elif (
                "next class" in message_lc
            ):

                intent = "next_class"

                is_confident = True

            elif (
                "schedule" in message_lc
                or "timetable" in message_lc
            ):

                intent = "todays_schedule"

                is_confident = True

            elif (
                "fee" in message_lc
                or "fees" in message_lc
            ):

                intent = "fee_info"

                is_confident = True

            # ==================================================
            # RESPONSE GENERATION
            # ==================================================

            if is_confident and intent in _VALID_INTENTS:

                # ==========================================
                # TEACHER LOCATION
                # ==========================================

                if intent == "teacher_location":

                    teacher_name = session.get("teacher_name")

                    if not teacher_name:

                        response = "Please mention teacher name."

                    else:

                        now_minutes = _time_to_minutes(current_time)

                        teacher = Teacher.query.filter(
                            Teacher.name.ilike(f"%{teacher_name}%")
                        ).first()

                        if not teacher:

                            response = (
                                f"No teacher found with name "
                                f"'{teacher_name}'."
                            )

                        else:

                            active_classes = TeacherTimetable.query.filter_by(
                                teacher_id=teacher.id,
                                day=current_day
                            ).all()

                            current_slot = [

                                t for t in active_classes

                                if (
                                    _time_to_minutes(t.start_time)
                                    <= now_minutes
                                    <= _time_to_minutes(t.end_time)
                                )
                            ]

                            if current_slot:

                                c = current_slot[0]

                                response = (
                                    f"📍 {teacher.name} is currently teaching "
                                    f"**{c.subject}** in Room {c.room} "
                                    f"from {c.start_time} to {c.end_time}."
                                )

                            else:

                                response = (
                                    f"{teacher.name} is not taking "
                                    f"any class right now."
                                )

                # ==========================================
                # OTHER INTENTS
                # ==========================================

                else:

                    response = _build_chatbot_response(
                        intent,
                        timetables,
                        current_day,
                        current_time
                    )

            else:

                response = (
                    "🤔 I can help you with your schedule, "
                    "classes, fees, or teacher locations."
                )

        # ==================================================
        # SAVE CHAT
        # ==================================================

        db.session.add(Chat(
            message=message,
            response=response,
            user_id=session['user_id']
        ))

        db.session.commit()

        return redirect(url_for('chatbot'))

    # ==================================================
    # GET CHAT HISTORY
    # ==================================================

    chats = Chat.query.filter_by(
        user_id=session['user_id']
    ).order_by(Chat.timestamp.asc()).all()

    return render_template(
        'chatbot.html',
        chats=chats
    )


# ==================================================
# CLEAR CHAT
# ==================================================

@app.route('/clear-chat')
@login_required
def clear_chat():
    Chat.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    return redirect(url_for('chatbot'))


# ==================================================
# ADMIN DASHBOARD
# ==================================================

@app.route('/admin')
@admin_required
def admin():
    return render_template(
        'admin.html',
        username=session['username'],
        courses=Course.query.all(),
        teachers=Teacher.query.all(),
        sections=Section.query.all(),
        students_count=User.query.filter_by(role='user').count(),
    )


# ==================================================
# FEES
# ==================================================

@app.route('/fees')
@admin_required
def fees():
    return render_template(
        'fees.html',
        courses=Course.query.all(),
        years=Year.query.all(),
        fees=FeeStructure.query.all(),
    )


@app.route('/add-fee-structure', methods=['POST'])
@admin_required
def add_fee():

    amount = request.form.get('amount')
    course_id = request.form.get('course_id')
    year_id = request.form.get('year_id')
    last_date = request.form.get('last_date', '')

    if not all([amount, course_id, year_id]):
        flash("All fee fields are required.", "error")
        return redirect(url_for('fees'))

    # ✅ CHECK DUPLICATE FEES
    existing_fee = FeeStructure.query.filter_by(
        course_id=course_id,
        year_id=year_id
    ).first()

    if existing_fee:
        flash("Fee structure already exists for this course and year.", "error")
        return redirect(url_for('fees'))

    db.session.add(
        FeeStructure(
            amount=int(amount),
            course_id=course_id,
            year_id=year_id,
            last_date=last_date
        )
    )

    db.session.commit()

    flash("Fee structure added successfully.", "success")

    return redirect(url_for('fees'))


# ==================================================
# COURSES
# ==================================================

@app.route('/courses')
@admin_required
def courses_page():
    return render_template('courses.html', courses=Course.query.all())


@app.route('/create-course-page')
@admin_required
def create_course_page():
    return render_template('create_course.html')


@app.route('/create-course', methods=['POST'])
@admin_required
def create_course():
    course_name = request.form.get('course', '').strip()
    
    if not course_name:
        flash("Course name is required.", "error")
        return redirect(url_for('create_course_page'))
    
    try:
        # Check if course already exists before adding
        existing_course = Course.query.filter_by(name=course_name).first()
        if existing_course:
            flash("Course already exists.", "error")
            return redirect(url_for('create_course_page'))
        
        db.session.add(Course(name=course_name))
        db.session.commit()
        flash("Course created successfully.", "success")
        return redirect(url_for('courses_page'))
        
    except Exception as e:
        db.session.rollback()
        flash("An error occurred while creating the course.", "error")
        return redirect(url_for('create_course_page'))


@app.route('/course/<int:course_id>')
@admin_required
def open_course(course_id):
    course = Course.query.get_or_404(course_id)
    return render_template(
        'course.html',
        username=session['username'],
        course=course,
        years=Year.query.filter_by(course_id=course_id).all(),
    )


# ==================================================
# YEARS
# ==================================================

@app.route('/create-year', methods=['POST'])
@admin_required
def create_year():

    year_name = request.form.get('year', '').strip()
    course_id = request.form.get('course_id')

    if not year_name or not course_id:
        flash("Year name and course are required.", "error")
        return redirect(url_for('courses_page'))

    # ✅ CHECK DUPLICATE YEAR
    existing_year = Year.query.filter_by(
        name=year_name,
        course_id=course_id
    ).first()

    if existing_year:
        flash("This year already exists for this course.", "error")
        return redirect(url_for('open_course', course_id=course_id))

    db.session.add(
        Year(
            name=year_name,
            course_id=course_id
        )
    )

    db.session.commit()

    flash("Year created successfully.", "success")

    return redirect(url_for('open_course', course_id=course_id))


@app.route('/year/<int:year_id>')
@admin_required
def open_year(year_id):
    year = Year.query.get_or_404(year_id)
    return render_template(
        'year.html',
        username=session['username'],
        year=year,
        sections=Section.query.filter_by(year_id=year_id).all(),
    )


# ==================================================
# SECTIONS
# ==================================================

@app.route('/create-section', methods=['POST'])
@admin_required
def create_section():

    section_name = request.form.get('section', '').strip()
    year_id = request.form.get('year_id')

    if not section_name or not year_id:
        flash("Section name and year are required.", "error")
        return redirect(url_for('courses_page'))

    # ✅ CHECK DUPLICATE SECTION
    existing_section = Section.query.filter_by(
        name=section_name,
        year_id=year_id
    ).first()

    if existing_section:
        flash("This section already exists.", "error")
        return redirect(url_for('open_year', year_id=year_id))

    db.session.add(
        Section(
            name=section_name,
            year_id=year_id
        )
    )

    db.session.commit()

    flash("Section created successfully.", "success")

    return redirect(url_for('open_year', year_id=year_id))


@app.route('/section/<int:section_id>')
@admin_required
def open_section(section_id):
    section    = Section.query.get_or_404(section_id)
    timetables = Timetable.query.filter_by(section_id=section_id).all()
    return render_template(
        'section.html',
        username=session['username'],
        section=section,
        timetables=timetables,
    )


# ==================================================
# TIMETABLE
# ==================================================

@app.route('/add-timetable', methods=['POST'])
@admin_required
def add_timetable():
    section_id = request.form.get('section_id')
    db.session.add(Timetable(
        subject    = request.form.get('subject', '').strip(),
        teacher    = request.form.get('teacher', '').strip(),
        start_time = request.form.get('start_time'),
        end_time   = request.form.get('end_time'),
        room       = request.form.get('room', '').strip(),
        day        = request.form.get('day'),
        section_id = section_id,
    ))
    db.session.commit()
    return redirect(url_for('open_section', section_id=section_id))


@app.route('/update/<int:id>', methods=['GET', 'POST'])
@admin_required
def update(id):
    timetable = Timetable.query.get_or_404(id)
    if request.method == 'POST':
        timetable.subject    = request.form.get('subject', '').strip()
        timetable.teacher    = request.form.get('teacher', '').strip()
        timetable.start_time = request.form.get('start_time')
        timetable.end_time   = request.form.get('end_time')
        timetable.room       = request.form.get('room', '').strip()
        timetable.day        = request.form.get('day')
        db.session.commit()
        return redirect(url_for('open_section', section_id=timetable.section_id))
    return render_template('updation.html', timetable=timetable)


@app.route('/delete/<int:id>')
@admin_required
def delete(id):
    timetable  = Timetable.query.get_or_404(id)
    section_id = timetable.section_id
    db.session.delete(timetable)
    db.session.commit()
    return redirect(url_for('open_section', section_id=section_id))


# ==================================================
# TEACHERS
# ==================================================

@app.route('/teachers')
@admin_required
def teachers_page():
    return render_template(
        'teachers.html',
        teachers=Teacher.query.all(),
        courses=Course.query.all(),
        sections=Section.query.all(),
    )


@app.route('/create-teacher-page')
@admin_required
def create_teacher_page():
    return render_template('create_teacher.html')


@app.route('/create-teacher', methods=['POST'])
@admin_required
def create_teacher():
    name = request.form.get('teacher', '').strip()
    if not name:
        flash("Teacher name is required.", "error")
        return redirect(url_for('create_teacher_page'))
    db.session.add(Teacher(name=name))
    db.session.commit()
    flash("Teacher added.", "success")
    return redirect(url_for('teachers_page'))


@app.route('/teacher/<int:teacher_id>')
@admin_required
def open_teacher(teacher_id):
    teacher    = Teacher.query.get_or_404(teacher_id)
    timetables = TeacherTimetable.query.filter_by(teacher_id=teacher_id).all()
    return render_template('teacher.html', teacher=teacher, timetables=timetables)


@app.route('/add-teacher-timetable', methods=['POST'])
@admin_required
def add_teacher_timetable():
    teacher_id = request.form.get('teacher_id')
    db.session.add(TeacherTimetable(
        subject    = request.form.get('subject', '').strip(),
        room       = request.form.get('room', '').strip(),
        day        = request.form.get('day'),
        start_time = request.form.get('start_time'),
        end_time   = request.form.get('end_time'),
        teacher_id = teacher_id,
    ))
    db.session.commit()
    return redirect(url_for('open_teacher', teacher_id=teacher_id))


# ==================================================
# LOGOUT
# ==================================================

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ==================================================
# AFTER-REQUEST: DISABLE CACHE
# ==================================================

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


# ==================================================
# CONTEXT PROCESSOR
# ==================================================

@app.context_processor
def inject_sidebar_data():
    return dict(
        courses  = Course.query.all(),
        teachers = Teacher.query.all(),
        sections = Section.query.all(),
    )


# ==================================================
# ERROR HANDLERS
# ==================================================

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return render_template('500.html'), 500


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=8000)S