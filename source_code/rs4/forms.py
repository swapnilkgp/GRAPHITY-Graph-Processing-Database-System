from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, PasswordField, SubmitField,DecimalField,SelectField, DateField, TimeField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError,NumberRange

from bs4 import BeautifulSoup

class Nodeform(FlaskForm):   
    node = IntegerField(label='Node:' ,validators=[DataRequired()])
    submit = SubmitField(label='Submit')

class Twoinputform(FlaskForm):   
    inp1 = IntegerField(label='Node:' ,validators=[DataRequired()])
    inp2 = IntegerField(label='Node:' ,validators=[DataRequired()])
    submit = SubmitField(label='Submit')