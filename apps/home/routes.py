# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from apps.home.models import get_eth_data
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound



@blueprint.route('/index')
@login_required
def index():

    eth_data = get_eth_data()
    end = len(eth_data)
    data_plot = eth_data.iloc[end-60:,]
    return render_template('home/home_v1.html', segment='index',data=eth_data)

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            pass

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
