import math
import os

from Reports import nowcasting_strat, benchmark, np
from report_monthly_util import PortfolioData, ReturnData, RiskData

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import defaultPageSize
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm


abs_root_path = 'C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Reports'
factsheet_save_path = 'C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Reports/Produced factsheets'
#%%

class PdfUtils:
    ''' utilities class to create a pdf '''

    def __init__(self):
        self.PAGE_DIM = {'width':defaultPageSize[0], 'height':defaultPageSize[1]}
        self.MARGINS = {'left':2.54*cm, 'right':self.PAGE_DIM['width'] - 2.54*cm, 'bottom':2.54*cm, 'top':self.PAGE_DIM['height'] - 2.54*cm}
        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        os.chdir(parent_dir)
        self.create_fonts()


    def create_fonts(self):
        ''' register the given fonts '''

        fonts_path = f'{abs_root_path}/Ressources/Fonts'
        pdfmetrics.registerFont(TTFont('Arial', f'{fonts_path}/arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial Bold', f'{fonts_path}/arial-bold.ttf'))
        pdfmetrics.registerFont(TTFont('Arial Italic', f'{fonts_path}/arial-italic.ttf'))
        pdfmetrics.registerFont(TTFont('Arial Nova Light', f'{fonts_path}/arial-nova-light.ttf'))
        self.par_style = ParagraphStyle('Normal', fontName='Arial', fontSize=10, alignment=4, spaceShrinkage=0.05, leading=13)
        self.color_black = colors.Color(red=(0), green=(0), blue=(0))
        self.color_white = colors.Color(red=(1), green=(1), blue=(1))
        self.color_grey = colors.Color(red=(128/255), green=(128/255), blue=(128/255))
        self.color_blue = colors.Color(red=(47/255), green=(84/255), blue=(150/255))
        self.color_light_grey = colors.Color(red=(242/255), green=(242/255), blue=(242/255))
        self.color_light_grey2 = colors.Color(red=(217/255), green=(217/255), blue=(217/255))
        self.color_light_blue = colors.Color(red=(140/255), green=(179/255), blue=(229/255))
        self.color_green = colors.Color(red=(76/255), green=(154/255), blue=(42/255))
        self.color_red = colors.Color(red=(210/255), green=(31/255), blue=(60/255))
        self.color_light_blue2 = colors.Color(red=(217/255), green=(226/255), blue=(243/255))



    def center_h_txt(self, txt, font, size):
        ''' give the x coord to center the text horizontally '''

        text_width = stringWidth(txt, font, size)
        return (self.PAGE_DIM['width'] - text_width) / 2




class PdfMaker(PdfUtils):
    ''' class to make the pdf '''

    def __init__(self, pdf_instance, intro_object):
        PdfUtils.__init__(self)
        self.pdf = pdf_instance
        self.colors = {'black':self.color_black, 'grey':self.color_grey, 'blue':self.color_blue,'light_grey': self.color_light_grey, 'light_grey_2': self.color_light_grey2,
                       'white': self.color_white, 'light_blue': self.color_light_blue, 'green':self.color_green, 'red':self.color_red, 'light_blue_2':self.color_light_blue2}
        self.date_dic = intro_object.dates_meta()


    def set_font(self, font, size):
        ''' set the given font '''

        self.pdf.setFont(font, size)


    def set_color(self, color_name):
        ''' set the given color '''

        self.pdf.setFillColor(self.colors[color_name])
        self.pdf.setStrokeColor(self.colors[color_name])


    def heading_1(self, text, y_coord):
        ''' add the heading to the pdf based on the text given as input '''

        self.set_font('Arial Nova Light', 16)
        self.set_color('blue')
        self.pdf.drawString(self.MARGINS['left'], y_coord, text)

        self.set_color('grey')
        self.pdf.rect(self.MARGINS['left'] - 1.5, y_coord-6, self.MARGINS['right'] - self.MARGINS['left'] +3, 1.5, fill=1, stroke=0)


    def heading_2(self, text, y_coord):
        ''' add an heading 2 style text to the pdf based on the text given as input '''

        self.set_font('Arial', 12)
        self.set_color('light_blue')
        self.pdf.drawString(self.MARGINS['left'], y_coord, text)


    def add_text(self, text, font, font_size, font_color, x_coord, y_coord):
        ''' add the given text to the pdf '''

        if x_coord == 'center':
            x_coord = self.center_h_txt(text, font, font_size)
        self.set_font(font, font_size)
        self.set_color(font_color)
        self.pdf.drawString(x_coord, y_coord, text)


    def add_paragraph(self, text, y_coord, x_coord=None):
        ''' add the given paragraph to the pdf '''

        if x_coord == None:
            x_coord = self.MARGINS['left']
        paragraph = Paragraph(text, self.par_style)
        paragraph.wrapOn(self.pdf, pdf_util.PAGE_DIM['width'] - pdf_util.MARGINS['left']*2, 100)
        paragraph.drawOn(self.pdf, x_coord, y_coord)


    def add_table(self, data, row_heights, col_widths, style, x_coord, y_coord):
        ''' add the given table to the pdf '''

        table = Table(data, rowHeights=row_heights, colWidths=col_widths)
        table.setStyle(style)
        table.wrapOn(self.pdf, 0, 0)
        table.drawOn(self.pdf, x_coord, y_coord)


    def draw_picture(self, name, graphic=False, **kwargs):
        ''' add a picture to the pdf '''

        if graphic != False:
            pic_path = f'{abs_root_path}/Ressources/Graphics'
        else:
            pic_path = f'{abs_root_path}/Ressources/Icons'
        self.pdf.drawImage(f'{pic_path}/{name}.png', **kwargs)


    def headers_setup(self, page_number):
        ''' set up the page headers and footers '''

        self.draw_picture(name='front_page', x=self.MARGINS['right'], y=self.MARGINS['top'], width=1.4*cm, height=1.4*cm, mask='auto')

        self.pdf.setFillColor(colors.Color(red=(217/255), green=(217/255), blue=(217/255)))
        p = self.pdf.beginPath()
        p.moveTo(0*cm, 1.5*cm)
        p.lineTo(0*cm, 0*cm)
        p.lineTo(self.PAGE_DIM['width'], 0*cm)
        p.lineTo(15.6*cm, 1.5*cm)
        p.lineTo(0*cm, 1.5*cm)
        self.pdf.drawPath(p, stroke=0, fill=1)

        self.pdf.setFillColor(colors.Color(red=(47/255), green=(85/255), blue=(151/255), alpha=0.8))
        p = self.pdf.beginPath()
        p.moveTo(13.76*cm, 0*cm)
        p.lineTo(self.PAGE_DIM['width'], 0*cm)
        p.lineTo(self.PAGE_DIM['width'], 2*cm)
        p.lineTo(13.76*cm, 0*cm)
        self.pdf.drawPath(p, stroke=0, fill=1)

        self.add_text(str(page_number).zfill(2), 'Arial Nova Light', 10, 'white', 560, 17)
        self.add_text(f'Fact sheet - {self.date_dic["period"]}', 'Arial Nova Light', 10, 'black', 38, 24)

        text = 'Trading profile'
        text = f'<font face="Arial Nova Light" size=10 color="blue"><u><link href="https://www.etoro.com/people/buzlclair123">{text}</link></u></font>'
        self.add_paragraph(text, x_coord=38, y_coord=9)


    def table_background(self, style, coords_list, color):
        ''' makes the cells from the given coord with a grey background. Coord list should be a list of list with each sublist being first coord tuple, second
            coord tuple '''

        for coord_1, coord_2 in coords_list:
            style.add('BACKGROUND', coord_1, coord_2, self.colors[color])
        return style


    def table_line(self, style, line_type, coords_list, color, width, line=None):
        ''' makes the cells from the given coord with a grey background. Coord list should be a list of list with each sublist being first coord tuple, second
            coord tuple '''

        line_dic = {'above':'LINEABOVE','below':'LINEBELOW', 'before':'LINEBEFORE', 'after':'LINEAFTER'}
        for coord_1, coord_2 in coords_list:
            style.add(line_dic[line_type], coord_1, coord_2, width, self.colors[color], 'squared', line)
        return style



def draw_circle(pdf_object, pdf_util, color, x, y, r):
    pdf_util.set_color(color)
    pdf_object.circle(x_cen=x, y_cen=y, r=r, stroke=0, fill=1)


def draw_rectangle(pdf_object, pdf_util, color, x, y, w, h):
    pdf_util.set_color(color)
    pdf_object.rect(x=x, y=y, width=w, height=h, stroke=0, fill=1)


def draw_indicator(pdf_object, coords_set):
    p = pdf_object.beginPath()
    p.moveTo(coords_set[0][0], coords_set[0][1])
    p.lineTo(coords_set[1][0], coords_set[1][1])
    p.lineTo(coords_set[2][0], coords_set[2][1])
    p.lineTo(coords_set[3][0], coords_set[3][1])
    p.lineTo(coords_set[4][0], coords_set[4][1])
    p.lineTo(coords_set[0][0], coords_set[0][1])
    pdf_object.drawPath(p, stroke=0, fill=1)


def rotate_point(x_coord, y_coord, pivot_coord_x, pivot_coord_y, angle):
    ''' Function to rotate a point (x, y) around a pivot (px, py) by an angle in radians '''

    s, c = math.sin(angle), math.cos(angle)

    # Translate the point to the origin
    x_coord -= pivot_coord_x
    y_coord -= pivot_coord_y

    new_x, new_y = x_coord * c - y_coord * s, x_coord * s + y_coord * c # Rotate the point

    # Translate the point back to its original position
    new_x += pivot_coord_x
    new_y += pivot_coord_y
    return new_x, new_y


def rotate_shape(coords, pivot_point, returns):
    angle = -2 * returns * math.pi # Calculate the angle based on the returns
    rotated_coords = [rotate_point(x, y, pivot_point[0], pivot_point[1], angle) for x, y in coords] # Rotate each point in the coords
    return rotated_coords


def draw_returns_gauge(pdf_object, pdf_util, center_x, center_y, radius, returns):
    ''' draw the returns gauge in page 1 of returns '''

    returns = min(0.25, max(-0.25, returns)) # keep the returns between -25% and 25% (if below / above, the rotation is the same as +-25%)

    start_arrow_x = center_x + radius*0.035
    start_arrow_y = center_y - radius*0.25
    coord_1 = (start_arrow_x, start_arrow_y)
    coord_2 = (start_arrow_x-radius*0.015, start_arrow_y+radius*1.3)
    coord_3 = (start_arrow_x-radius*0.035, start_arrow_y+radius*1.5)
    coord_4 = (start_arrow_x-radius*0.055, start_arrow_y+radius*1.3)
    coord_5 = (start_arrow_x-radius*0.07, start_arrow_y)
    coords = [coord_1, coord_2, coord_3, coord_4, coord_5]
    pivot_point = (center_x, center_y)
    coords_rotated = rotate_shape(coords, pivot_point, returns)

    draw_circle(pdf_object, pdf_util, 'light_grey_2', center_x, center_y, radius)
    draw_circle(pdf_object, pdf_util, 'white', center_x, center_y, radius * 0.8)
    draw_rectangle(pdf_object, pdf_util, 'white', center_x-radius, center_y-radius, radius*2, radius)

    draw_rectangle(pdf_object, pdf_util, 'grey', center_x-0.5, center_y+radius*0.8, 1, radius*0.1)


    extent = (returns/0.25)*90 * (returns>0) - (returns/-0.25)*90 * (returns<0)
    if returns >0:
        pdf.setFillColorRGB(0.298, 0.604, 0.165, 0.2)
    else:
        pdf.setFillColorRGB(0.824, 0.122, 0.235, 0.2)
    pdf.wedge(center_x-radius*0.8, center_y-radius*0.8, center_x+radius*0.8, center_y+radius*0.8, startAng=90, extent=-extent, stroke=0, fill=1)

    pdf_util.add_text(text='0%', font='Arial', font_size=8, font_color='grey', x_coord=center_x - stringWidth('0%', 'Arial', 8)/2, y_coord=center_y+radius*0.6)

    pdf_util.set_color('light_blue')
    draw_indicator(pdf_object, coords_rotated)

    draw_circle(pdf, pdf_util, 'grey', center_x, center_y, radius*0.125)





def page_1(pdf_object, pdf_instance, intro_object):
    ''' creates the first page of the pdf '''

    date_dic = intro_object.dates_meta()

    pdf_instance.draw_picture(name='front_page', x=pdf_instance.PAGE_DIM['width']-5*cm, y=pdf_instance.PAGE_DIM['height']-5*cm, width=5*cm, height=5*cm, mask='auto')
    pdf_instance.add_text('Investment Strategy', 'Arial Bold', 28, 'black', 'center', 695.5)
    pdf_instance.add_text(f'{date_dic["period"]} Factsheet', 'Arial', 18, 'grey', 'center', 673)
    pdf_instance.heading_1('INTRODUCTION', 593)

    pdf_instance.heading_2('Factsheet Strategy', 552)
    text = '''This investment strategy aims to generate capital growth by picking rising stocks over a pre-selected investment universe. The objective is typically to
            outperform a retail investment scenario where an individual would buy and hold the investment universe. The rising stocks identification is performed
            through technical analysis with a long-only and monthly rebalancing investment scheme.'''
    pdf_instance.add_paragraph(text, 488)
    data= [['General Information', ''], ['Asset class', 'Equity'], ['Management style', 'Active'], ['Investment currency', 'USD'],
           ['Number of months since live', date_dic["months since live"]], ['Performance ranking since live', f'{intro_object.returns_ranking()} / {date_dic["months since live"]}']]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(-1,0),(-1,-1),'CENTER'), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (0,0), 11),
                        ('FONTSIZE', (0,1), (-1,-1), 9), ('LINEBELOW', (0,0), (-1,0), 1.5, pdf_util.colors['blue'], 'squared'),
                        ('LINEBELOW', (0,1), (-1,1), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)), ('LINEBELOW', (0,2), (-1,2), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,3), (-1,3), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)), ('LINEBELOW', (0,4), (-1,4), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,5), (-1,5), 0.5, pdf_instance.colors['grey'], 'squared'),
                        ('BACKGROUND',(0,1),(0,-1), pdf_instance.colors['light_grey'])])
    pdf_instance.add_table(data, row_heights=0.82*cm, col_widths=(4.9*cm, 3.8*cm), style=style, x_coord=(pdf_instance.PAGE_DIM['width']-8.7*cm)/2, y_coord=336)

    pdf_instance.heading_2('Factsheet Coverage', 300.5)
    text = f''' This factsheet covers the performance of the strategy over {date_dic["business days"]} business days from {date_dic["range"]}. The S&amp;P 500 performance for the past 100 days below
    provides hints on the overall stock market sentiment during the period. '''
    pdf_instance.add_paragraph(text, 250.5)
    pdf_instance.draw_picture(name='sp_eco_cdt', graphic=True, x=(pdf_instance.PAGE_DIM['width']-10*cm)/2, y=82, width=10*cm, height=5.52*cm, mask='auto')

    pdf_object.showPage()



def returns_table_formatting(style, check_data):
    ''' do the conditional formatting of the table of monthly returns present in the RETURNS part '''

    check_data = returns_data.monthly_returns_table()
    for year, values in enumerate(check_data[1:,1:]):
        for month, value in enumerate(values):
            if value == '':
                pass
            elif value < 0:
                style.add('TEXTCOLOR', (1+month, 1+year), (1+month, 1+year), 'red')
            elif value > 0:
                style.add('TEXTCOLOR', (1+month, 1+year), (1+month, 1+year), 'green')

        if check_data[year+1][0] < strategy_data.implem_date.year:
            style.add('BACKGROUND',(1,year+1),(-2,year+1), pdf_util.colors['light_grey'])
        elif check_data[year+1][0] == strategy_data.implem_date.year:
            for month, value in enumerate(values):
                if month+1 < strategy_data.implem_date.month:
                    style.add('BACKGROUND',(month+1,year+1),(month+1,year+1), pdf_util.colors['light_grey'])
                elif month+1 == strategy_data.implem_date.month and type(check_data[year+1,month+1])==float:
                    check_data[year+1,month+1] = f'{check_data[year+1,month+1]}*'
    return style, check_data


def risk_indic_table_formatting(style, risk_data):
    ''' do the conditional formatting of the table of risk indicator present in the RISK part '''

    strat_std = risk_data.std_risk['Risk indic']
    month_calc = list(map(lambda x: abs(x - risk_data.portfolio_instance.dates_list[-1]), strat_std.index))
    current_month = month_calc.index(min(month_calc))
    risk_indic = math.ceil(risk_data.std_risk['Risk indic'][current_month])
    col_index = risk_indic-1

    style.add('SPAN', (col_index,0), (col_index,1))
    style.add('LINEBELOW', (col_index,1), (col_index,1), 1, pdf_util.colors['light_blue_2'])
    style.add('LINEABOVE', (col_index,1), (col_index,1), 1, pdf_util.colors['light_blue_2'])
    style.add('BOX', (col_index,0), (col_index,-1), 0.5, pdf_util.colors['blue'], 'squared')
    style.add('INERGRID', (col_index,0), (col_index,-1), 0.5, pdf_util.colors['blue'], 'squared')
    style.add('FONTNAME', (col_index,0), (col_index,-1), 'Arial Bold')
    style.add('FONTSIZE', (col_index,0), (col_index,-1), 20)
    style.add('TEXTCOLOR', (col_index,0), (col_index,-1), pdf_util.colors['black'])
    style.add('BACKGROUND', (col_index,0), (col_index,-1), pdf_util.colors['light_blue_2'])
    style.add('VALIGN',(col_index,0),(col_index,1),'TOP')
    return style



def returns_part_1(pdf_object, pdf_instance, returns_instance):
    ''' creates the first part of the return topic of the document '''

    key_numbers = returns_data.key_numbers
    pdf_instance.headers_setup(1)
    pdf_instance.heading_1('RETURNS', 754)

    part_coord = 694.5
    pdf_instance.heading_2('Last period key numbers', part_coord)
    text = ''' Below are the key statistics of the strategy over the last investment period. '''
    pdf_instance.add_paragraph(text, part_coord-23.5)

    draw_returns_gauge(pdf_object, pdf_instance, center_x=220, center_y=part_coord-195, radius=60, returns=key_numbers["return"])


    data = [['', '% of profitable days'], ['', f'{"{:.0%}".format(key_numbers["pct profitable"])}'], ['', 'Monthly high (starting at 100)'],
            ['', f'{"{:.1f}".format(key_numbers["high"])}'], ['', 'Monthly low (starting at 100)'], ['', f'{"{:.1f}".format(key_numbers["low"])}']]
    style = TableStyle([('VALIGN',(-1,-1),(-1,-1),'TOP'), ('VALIGN',(-1,1),(-1,1),'TOP'), ('VALIGN',(-1,3),(-1,3),'TOP'), ('VALIGN',(0,2),(0,2),'TOP'),
                        ('VALIGN',(0,1),(0,1),'BOTTOM'), ('VALIGN',(-1,0),(-1,0),'BOTTOM'), ('VALIGN',(-1,2),(-1,2),'BOTTOM'), ('VALIGN',(-1,4),(-1,4),'BOTTOM'),
                        ('ALIGN',(0,2),(0,-1),'CENTER'), ('ALIGN',(-1,1),(-1,1),'CENTER'), ('ALIGN',(-1,3),(-1,3),'CENTER'), ('ALIGN',(-1,-1),(-1,-1),'CENTER'),
                        ('ALIGN',(0,1),(0,1),'LEFT'), ('ALIGN',(-1,0),(-1,0),'LEFT'), ('ALIGN',(-1,2),(-1,2),'LEFT'), ('ALIGN',(-1,4),(-1,4),'LEFT'),
                        ('TEXTCOLOR',(-1,1),(-1,1), pdf_instance.colors['blue']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE',(0,2),(0,2),72),
                        ('FONTSIZE',(0,1),(0,1),18), ('FONTSIZE',(-1,0),(-1,0),10), ('FONTSIZE',(-1,2),(-1,2),10), ('FONTSIZE',(-1,4),(-1,4),10),
                        ('FONTSIZE',(-1,1),(-1,1),28), ('FONTSIZE',(-1,3),(-1,3),28), ('FONTSIZE',(-1,-1),(-1,-1),28), ('BACKGROUND',(1,0),(-1,-1), pdf_instance.colors['light_grey']),
                        ('LINEABOVE', (0,0), (-1,0), 0.75, pdf_instance.colors['black'], 'squared'), ('LINEBELOW', (0,-1), (-1,-1), 0.75, pdf_instance.colors['black'], 'squared'),
                        ('LINEBELOW', (-1,1), (-1,1), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)), ('LINEBELOW', (-1,3), (-1,3), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2))])
    if key_numbers['return'] > 0:
        style.add('TEXTCOLOR',(0,2),(0,-1), pdf_util.colors['green'])
    else:
        style.add('TEXTCOLOR',(0,2),(0,-1), pdf_util.colors['red'])
    pdf_instance.add_table(data, row_heights=(0.7*cm,1.4*cm,0.7*cm,1.4*cm,0.7*cm,1.4*cm), col_widths=(8.6*cm, 5.4*cm), style=style, x_coord=(pdf_util.PAGE_DIM['width']-14*cm)/2, y_coord=part_coord-214)
    arrow_dim = {'h1':9, 'h2':8, 'w1':4,'w2':5}
    arrow_start = (468,551)
    pdf_object.setFillColor(colors.Color(red=(47/255), green=(84/255), blue=(150/255), alpha=0.8))
    p = pdf_object.beginPath()
    p.moveTo(arrow_start[0], arrow_start[1])
    p.lineTo(arrow_start[0], arrow_start[1] + arrow_dim['h1'])
    p.lineTo(arrow_start[0] + arrow_dim['w1'], arrow_start[1] + arrow_dim['h1'])
    p.lineTo(arrow_start[0] - arrow_dim['w2']/2, arrow_start[1] + arrow_dim['h1'] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] - arrow_dim['w1'] - arrow_dim['w2'], arrow_start[1] + arrow_dim['h1'])
    p.lineTo(arrow_start[0] - arrow_dim['w2'], arrow_start[1] + arrow_dim['h1'])
    p.lineTo(arrow_start[0] - arrow_dim['w2'], arrow_start[1])
    p.lineTo(arrow_start[0], arrow_start[1])
    pdf_object.drawPath(p, stroke=0, fill=1)

    arrow_start = (377,549-2.1*cm)
    pdf_object.setFillColor(colors.Color(red=(128/255), green=(128/255), blue=(128/255), alpha=0.8))
    p = pdf_object.beginPath()
    p.moveTo(arrow_start[0], arrow_start[1])
    p.lineTo(arrow_start[0] + arrow_dim['w1'] + 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] + 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] + 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h1'] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] - 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h1'] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] - 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h2'])
    p.lineTo(arrow_start[0] - arrow_dim['w1'] - 0.5*arrow_dim['w2'], arrow_start[1] + arrow_dim['h2'])
    p.lineTo(arrow_start[0], arrow_start[1])
    pdf_object.drawPath(p, stroke=0, fill=1)

    pdf_instance.add_text('Monthly return', 'Arial', font_size=10, font_color='black', x_coord=105, y_coord=part_coord-51)
    if key_numbers['return'] >= 0:
        color = 'green'
    else:
        color = 'red'
    text = f'{key_numbers["return"]:.1%}'
    pdf_instance.add_text(text, 'Arial', font_size=50, font_color=color, x_coord=225-stringWidth(text, 'Arial', 50)/2, y_coord=part_coord-104)


    part_coord = 410
    pdf_instance.heading_2('Strategy compared to benchmark', part_coord)
    text = ''' The graphic below compare the performance of the strategy to an hypothetical buy and hold benchmark over the last period. The y axis
    scale the performance analysis to an initial $100 investment. This analysis is performed over the last 100 business days (if the sample allows it). '''
    pdf_instance.add_paragraph(text, part_coord-50)
    pdf_instance.draw_picture(name='Portfolio performance - Strategy vs Buy and Hold', graphic=True, x=(pdf_instance.PAGE_DIM['width']-10*cm)/2, y=part_coord-220, width=10*cm, height=5.59*cm, mask='auto')
    text = ''' The performances were computed starting at the strategy go-live date to have a fair representation of the stocks units (especially for
    the Buy and Hold approach). The performance was then rescaled at the beginning of the last 100 business days to facilitate the comparison between both approaches. '''
    pdf_instance.add_paragraph(text, part_coord-267)

    pdf_object.showPage()



def returns_part_2(pdf_object, pdf_instance, returns_instance):
    ''' creates the first part of the return topic of the document '''

    returns_table = returns_instance.monthly_returns_table()
    pdf_instance.headers_setup(2)

    part_coord = 759
    pdf_instance.heading_2('Performance due to rebalancing', part_coord)
    text = ''' The following plot is similar to the previous performance comparison between the benchmark and the strategy. In this case, we compare
    the performance of the strategy to the performance that the strategy would have obtained if no rebalancing was performed at the beginning of the period.'''
    pdf_instance.add_paragraph(text, part_coord-50)
    pdf_instance.draw_picture(name='Portfolio performance - Strategy vs Strategy without EOM rebalance', graphic=True, x=(pdf_instance.PAGE_DIM['width']-10*cm)/2, y=part_coord-221, width=10*cm, height=5.59*cm, mask='auto')
    text = ''' In other words, the plot highlights the value added by rebalancing according to the strategy signals at the beginning of the period.'''
    pdf_instance.add_paragraph(text, part_coord-259)

    part_coord = 446
    pdf_instance.heading_2('Monthly Returns', part_coord)

    text = f''' The table below presents the monthly returns of the strategy for the period from January 2020 to {returns_instance.portfolio_instance.dates_list[-1].strftime("%B %Y")}.
    It provides an overview of the strategy’s performance on a month-by-month basis.'''
    pdf_instance.add_paragraph(text, part_coord-37)

    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(0,0),(-1,-1),'CENTER'),
                        ('TEXTCOLOR', (0,0), (-1,0), pdf_instance.colors['grey']), ('FONTNAME', (0,0), (-1,0), 'Arial Nova Light'), ('FONTSIZE', (0,0), (-1,0), 10),
                        ('TEXTCOLOR', (0,1), (0,-1), pdf_instance.colors['black']), ('FONTNAME', (0,1), (-1,-1), 'Arial'), ('FONTSIZE', (0,1), (0,-1), 10),
                        ('FONTSIZE', (1,1), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 0.5, pdf_instance.colors['blue'], 'squared'), ('LINEBELOW', (0,0), (-1,0), 0.5, pdf_instance.colors['blue'], 'squared'),
                        ('LINEBELOW', (0,-1), (-1,-1), 0.5, pdf_instance.colors['blue'], 'squared'), ('LINEBEFORE', (-1,1), (-1,-1), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2))])
    style, data = returns_table_formatting(style, returns_table)
    data = data.tolist()
    pdf_instance.add_table(data, row_heights=0.6*cm, col_widths=(1.14*cm), style=style, x_coord=pdf_instance.MARGINS['left'], y_coord=part_coord-133)
    txt = '*Strategy went live in July 2023'
    pdf_instance.add_text(txt, font='Arial Italic', font_size=8, font_color='grey', x_coord=pdf_instance.MARGINS['right'] - stringWidth(txt, 'Arial Italic', 8), y_coord=part_coord-144)
    txt = 'Backtesting performance'
    pdf_instance.add_text(txt, font='Arial Italic', font_size=9, font_color='black', x_coord=pdf_instance.MARGINS['right'] - stringWidth(txt, 'Arial Italic', 9), y_coord=part_coord-157)
    pdf_instance.set_color('light_grey')
    pdf_object.rect(x=pdf_instance.MARGINS['right'] - stringWidth(txt, 'Arial Italic', 9) - 26, y=part_coord-158, width=20, height=8, fill=1, stroke=0)
    text = ''' Please note that the returns displayed before the strategy's go-live date were estimated using historical
    data and backtesting. They do not represent actual returns generated by the strategy during that period.'''
    pdf_instance.add_paragraph(text, part_coord-208)

    pdf_object.showPage()



def returns_part_3(pdf_object, pdf_instance, returns_instance):
    ''' creates the second part of the return topic of the document '''

    sr_distrib = returns_instance.sharpe_distribution()

    pdf_instance.headers_setup(3)
    pdf_instance.heading_2('Last Period Performance', 759)

    text = ''' This distribution plot compares the performance of the strategy to a random sample over the last period. The Sharpe ratio is annualized and computed with an estimated risk-free rate of 0.
    <br/>The sample is composed of 30’000 randomly constructed portfolios. Each portfolio is built by randomly selecting which stocks from the investment universe will be picked out.
    For every stock, a "0" or "1" is randomly chosen to decide if that particular stock is included in the portfolio. This portfolio construction process is repeated 30'000 to create the sample. '''
    pdf_instance.add_paragraph(text, 670)
    pdf_instance.draw_picture(name='strategy_returns_ditrib', graphic=True, x=pdf_instance.MARGINS['left'], y=pdf_instance.MARGINS['top']-275, width=10*cm, height=5.55*cm, mask='auto')
    pdf_instance.add_text('Key Highlights', font='Arial', font_size=10, font_color='black', x_coord=pdf_instance.MARGINS['right'] - 40 - stringWidth('Key Highlights', 'Arial', 10), y_coord=pdf_instance.MARGINS['top'] - 120)

    sample_avg_sr = '{:.2f}'.format(np.mean(sr_distrib))
    percentiles = ['{:.2f}'.format(np.percentile(sr_distrib, x)) for x in [25,50,75]]
    strategy_sr = '{:.2f}'.format(returns_instance.last_sharpe())
    data= [["30'000", 'Sample size'], [sample_avg_sr, 'Sample avg \nSharpe ratio'], [f'{percentiles[0]}, {percentiles[1]}, {percentiles[2]}', 'Percentile \n(25, 50, 75)'], [strategy_sr, 'Strategy \nSharpe ratio']]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(-1,0),(-1,-1),'LEFT'), ('ALIGN',(0,0),(0,-1),'CENTER'),
                        ('TEXTCOLOR', (-1,0),(-1,-1), pdf_instance.colors['grey']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 1, pdf_instance.colors['blue'], 'squared'), ('LINEABOVE', (0,1), (-1,1), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEABOVE', (0,2), (-1,2), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)), ('LINEABOVE', (0,3), (-1,3), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,3), (-1,3), 1, pdf_instance.colors['black'], 'squared'), ('BACKGROUND',(0,0),(0,-1), pdf_instance.colors['light_grey'])])
    pdf_instance.add_table(data, row_heights=1.25*cm, col_widths=(2.72*cm, 2.72*cm), style=style, x_coord=pdf_instance.MARGINS['right']-5.42*cm, y_coord=pdf_instance.MARGINS['top'] - 267.5)

    text = ''' This Sharpe ratio comparison provides insights on the strategy’s performance over the last period given the investment universe. For instance, if the strategy obtains a percentile value of 70%, it means
    that in terms of risk-adjusted returns, the strategy performed better than 70% of the portfolios possible within the investment universe. '''
    pdf_instance.add_paragraph(text, 435)


    part_start_coord = 382
    pdf_instance.heading_2('Historical Performance', part_start_coord)
    text = ''' The following graphic illustrates the value of a $100 investment started in January 2020 with the strategy compared to the same investment made on
    the benchmark initiated at the same starting date.<br/>The benchmark used in comparison consists in an "equiweighted Buy and Hold portfolio" comprising all stocks within the investment
    universe and keeping the initial positions open without rebalancing throughout the evaluation period. '''
    pdf_instance.add_paragraph(text, part_start_coord-76)

    pdf_instance.draw_picture(name='strategy_investment_value', graphic=True, x=pdf_instance.MARGINS['right']-10*cm, y=part_start_coord-250, width=10*cm, height=5.55*cm, mask='auto')
    pdf_instance.add_text('Key Highlights', font='Arial', font_size=10, font_color='black', x_coord=pdf_instance.MARGINS['left'] + 40, y_coord=part_start_coord-100)

    start_date = strategy_data.performance_wealth['Strategy'].index[0].strftime('%d.%m.%Y')
    total_return = round(100 * strategy_data.performance_wealth['Strategy'][-1] / strategy_data.performance_wealth['Strategy'][0], 2)
    outperf = round(total_return - 100 * strategy_data.performance_wealth['Benchmark'][-1] / strategy_data.performance_wealth['Benchmark'][0], 2)
    data = [['Starting date', start_date], ['Initial investment', '$100'], ['Final value', f'${total_return}'], ['Outperformance', f'${outperf}']]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(-1,0),(-1,-1),'CENTER'), ('ALIGN',(0,0),(0,-1),'RIGHT'),
                        ('TEXTCOLOR', (0,0), (0,-1), pdf_instance.colors['grey']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 1, pdf_instance.colors['blue'], 'squared'), ('LINEABOVE', (0,1), (-1,1), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEABOVE', (0,2), (-1,2), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)), ('LINEABOVE', (0,3), (-1,3), 0.5, pdf_instance.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,3), (-1,3), 1, pdf_instance.colors['black'], 'squared'), ('BACKGROUND',(1,0),(-1,-1), pdf_instance.colors['light_grey'])])
    pdf_instance.add_table(data, row_heights=1.25*cm, col_widths=(2.72*cm, 2.72*cm), style=style, x_coord=pdf_instance.MARGINS['left'], y_coord=part_start_coord-250)
    txt = 'Backtesting data'
    pdf_instance.add_text(txt, font='Arial Italic', font_size=8, font_color='grey', x_coord=pdf_instance.MARGINS['right'] - stringWidth(txt, 'Arial Italic', 8), y_coord=part_start_coord-254)
    pdf_instance.set_color('light_grey')
    pdf_object.rect(x=pdf_instance.MARGINS['right'] - stringWidth(txt, 'Arial Italic', 8) - 26, y=part_start_coord-255, width=20, height=8, fill=1, stroke=0)

    pdf_object.showPage()



def risk_part_1(pdf, pdf_util, risk_data):
    ''' creates the first part of the risk topic of the document '''

    pdf_util.headers_setup(4)
    pdf_util.heading_1('RISK', 754.5)

    pdf_util.heading_2('Risk Metric', 695)
    text = ''' The risk metric is a measure based on the strategy’s weekly annualized volatility. It ranges from 1 to 10 with 1 being the lowest possible risk
    rating for the strategy and 10 being the highest. The monthly average of the volatility gives the indicator value below.'''
    pdf_util.add_paragraph(text, 645)
    data= [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['','','','','','','','','','']]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('BOX', (0,1), (-1,1), 0.5, pdf_util.colors['grey'], 'squared'), ('INNERGRID', (0,1), (-1,1), 0.5, pdf_util.colors['grey'], 'squared'),
                        ('TEXTCOLOR', (0,0),(-1,0), pdf_util.colors['white']), ('BACKGROUND',(0,1),(-1,1), pdf_util.colors['light_grey'])])
    style = risk_indic_table_formatting(style, risk_data)
    pdf_util.add_table(data, row_heights=(0.3*cm,0.61*cm, 0.3*cm), col_widths=(1.5*cm), style=style, x_coord=(pdf_util.PAGE_DIM['width']-15*cm)/2, y_coord=597)
    pdf_util.draw_picture(name='left_arrow_head', graphic=False, x=pdf_util.MARGINS['left']+15, y=588, width=6, height=5, mask='auto')
    pdf_util.add_text('lower risk', font='Arial Italic', font_size=9, font_color='black', x_coord=pdf_util.MARGINS['left']+24, y_coord=587.5)
    pdf_util.draw_picture(name='right_arrow_head', graphic=False, x=pdf_util.MARGINS['right']-21, y=588, width=6, height=5, mask='auto')
    pdf_util.add_text('higher risk', font='Arial Italic', font_size=9, font_color='black', x_coord=pdf_util.MARGINS['right']-24-stringWidth('higher risk', 'Arial Italic', 9), y_coord=587.5)
    text = ''' The thresholds applied to convert the volatility into the risk indicator are the following: '''
    pdf_util.add_paragraph(text, 549)
    data= [['Annualized \nweekly volatility', '3%', '6%', '9%', '12.5%', '16%', '21%', '27%', '34%', '43%', '55%'],
           ['Risk indicator', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(0,0),(0,-1),'LEFT'), ('ALIGN',(1,0),(-1,-1),'CENTER'),
                        ('TEXTCOLOR', (1,0),(-1,0), pdf_util.colors['grey']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 0.5, pdf_util.colors['blue'], 'squared'), ('LINEBELOW', (0,1), (-1,1), 0.5, pdf_util.colors['blue'], 'squared'),
                        ('LINEBELOW', (0,0), (-1,0), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)), ('BACKGROUND',(0,0),(0,-1), pdf_util.colors['light_grey'])])
    pdf_util.add_table(data, row_heights=0.9*cm, col_widths=(2.6*cm,cm,cm,cm,cm,cm,cm,cm,cm,cm,cm), style=style, x_coord=(pdf_util.PAGE_DIM['width']-12.6*cm)/2, y_coord=485)

    pdf_util.heading_2('Historical Volatility', 410)
    text = ''' The chart below represents the strategy's annualized volatility over the last months. Each column in the graph
    corresponds to the annualized volatility for a specific month. For each month, it represents how much the strategy's performance would have varied over
    a year if the strategy’s performance for a given month was to persist in the same manner throughout the entire year. '''
    pdf_util.add_paragraph(text, 347)
    pdf_util.draw_picture(name='strategy_std_risk', graphic=True, x=(pdf_util.PAGE_DIM['width']-10*cm)/2, y=178, width=10*cm, height=5.54*cm, mask='auto')
    text = ''' This analysis provides insights into the strategy's risk exposure and how is evolution over time. Higher values indicate months with increased variation in the portfolio's returns,
    while lower points represent periods of relative stability. '''
    pdf_util.add_paragraph(text, 126.5)

    pdf.showPage()



def risk_part_2(pdf, pdf_util, risk_data):
    ''' creates the second part of the risk topic of the document '''

    pdf_util.headers_setup(5)

    pdf_util.heading_2('Yearly Drawdowns', 759)
    text = ''' This chart represents the performance drawdowns of the strategy in comparison to the Buy and hold benchmark over the last year, both based on
    the same investment universe. Performance drawdowns represent the decline in value from a previous peak, indicating periods when the strategy
    experienced losses. In other word, it depicts the straight consecutive losses that a potential investor on this strategy has experienced. '''
    pdf_util.add_paragraph(text, 683)
    pdf_util.draw_picture(name='drawdown_risk_plot', graphic=True, x=pdf_util.MARGINS['left'], y=514.5, width=10*cm, height=5.48*cm, mask='auto')
    pdf_util.add_text('Key Highlights', font='Arial', font_size=10, font_color='black', x_coord=414, y_coord=657)
    max_dd = abs(risk_data.drawdown.min(axis=0) * 100)
    avg_dd = abs(risk_data.drawdown.mean(axis=0) * 100)
    data= [[f"{'{:.2f}'.format(max_dd['Strategy'])}%", 'Strategy \nMax drawdown'], [f"{'{:.2f}'.format(avg_dd['Strategy'])}%", 'Strategy \nAvg drawdown'],
           [f"{'{:.2f}'.format(max_dd['Benchmark'])}%", 'Buy and hold \nMax drawdown'], [f"{'{:.2f}'.format(avg_dd['Benchmark'])}%", 'Buy and hold \nAvg drawdown']]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(-1,0),(-1,-1),'LEFT'), ('ALIGN',(0,0),(0,-1),'CENTER'),
                        ('TEXTCOLOR', (-1,0),(-1,-1), pdf_util.colors['grey']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 1, pdf_util.colors['blue'], 'squared'), ('LINEABOVE', (0,1), (-1,1), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)),
                        ('LINEABOVE', (0,2), (-1,2), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)), ('LINEABOVE', (0,3), (-1,3), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,3), (-1,3), 1, pdf_util.colors['black'], 'squared'), ('BACKGROUND',(0,0),(0,-1), pdf_util.colors['light_grey'])])
    pdf_util.add_table(data, row_heights=1.15*cm, col_widths=(2.72*cm, 2.72*cm), style=style, x_coord=pdf_util.MARGINS['right']-5.42*cm, y_coord=520)

    std_distrib = risk_data.std_distribution()
    pdf_util.heading_2('Annualized Volatility Simulation', 448)
    text = ''' This distribution plot compares the annualized volatility of the strategy to a random sample over the last period. Similarly to the Sharpe ratio
    distribution plot, the random sample is composed of 30’000 portfolios for which each composition is randomly selected within the investment universe. '''
    pdf_util.add_paragraph(text, 398)
    pdf_util.draw_picture(name='strategy_std_ditrib', graphic=True, x=pdf_util.MARGINS['right']-10*cm, y=228, width=10*cm, height=5.5*cm, mask='auto')
    pdf_util.add_text('Key Highlights', font='Arial', font_size=10, font_color='black', x_coord=pdf_util.MARGINS['left'] + 45.5, y_coord=371.5)
    sample_avg_std = '{:.2f}'.format(np.mean(std_distrib))
    percentiles = ['{:.2f}'.format(np.percentile(std_distrib, x)) for x in [25,50,75]]
    strategy_std = '{:.2f}'.format(risk_data.last_std())
    data = [['Sample size', "30'000"], ['Sample avg \nVolatility', sample_avg_std], ['Percentile \n(25, 50, 75)', f'{percentiles[0]}, {percentiles[1]}, {percentiles[2]}'], ['Strategy Volatility', strategy_std]]
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(-1,0),(-1,-1),'CENTER'), ('ALIGN',(0,0),(0,-1),'RIGHT'),
                        ('TEXTCOLOR', (0,0), (0,-1), pdf_util.colors['grey']), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (0,0), (-1,0), 1, pdf_util.colors['blue'], 'squared'), ('LINEABOVE', (0,1), (-1,1), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)),
                        ('LINEABOVE', (0,2), (-1,2), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)), ('LINEABOVE', (0,3), (-1,3), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)),
                        ('LINEBELOW', (0,3), (-1,3), 1, pdf_util.colors['black'], 'squared'), ('BACKGROUND',(1,0),(-1,-1), pdf_util.colors['light_grey'])])
    pdf_util.add_table(data, row_heights=1.15*cm, col_widths=(2.72*cm, 2.72*cm), style=style, x_coord=pdf_util.MARGINS['left'], y_coord=234.5)
    text = ''' This volatility comparison provides insights on the strategy’s performance over the last period given the investment universe. A low volatility is generally
    wished, so as opposed to the previous distribution plot, we prefer a low percentile for this graph (if the strategy obtains a percentile value of 30%, it means
    that in terms of volatility, the strategy was more stable than 70% of the portfolios possible given the investment universe). '''
    pdf_util.add_paragraph(text, 150)

    pdf.showPage()



def risk_part_3(pdf, pdf_util, risk_data):
    ''' creates the third part of the risk topic of the report '''

    pdf_util.headers_setup(6)

    pdf_util.heading_2('Yearly Rolling Beta', 759)
    text = ''' This plot explains the impacts of changes in the stock market on the investment strategy performance. The analysis was performed using the S&amp;P 500 as a market proxy. Beta is a metic that quantifies how much an
    a portfolio changes in reaction to a market movement. If beta is greater (resp. lower) than 1, the investment is more (resp. less) volatile than the market. For instance,
    a beta of 1.2 indicates that when the market goes up or down by 1%, our investment will go up or down by 1.2%. It's a way to understand how the strategy reacts to market changes. '''
    pdf_util.add_paragraph(text, 670)
    pdf_util.draw_picture(name='rolling_12m_beta', graphic=True, x=(pdf_util.PAGE_DIM['width']-10*cm)/2, y=500.5, width=10*cm, height=5.52*cm, mask='auto')
    text = ''' By using a rolling 12-month beta calculation, this graph depicts the evolution of the strategy's relationship with the S&amp;P 500 over time. The varying beta values highlight
    periods when the strategy showed increased or decreased correlation with the market. This analysis provides insights to assess how the strategy has historically adapted to diverse market conditions.
    Analyzing these fluctuations helps in understanding how the strategy performs in various market conditions, offering a perspective on its risk exposure and potential diversification benefits
    when compared with the broader market movements represented by the S&amp;P 500. '''
    pdf_util.add_paragraph(text, 398)


    pdf_util.heading_2('Historical Correlation Matrix', 344.5)

    def _clean_data():
        ''' returns the ready to use data for the correlation matrix '''

        matrix = risk_data.corr_matrix()
        triangular_matrix = np.triu(matrix)
        triangular_matrix = np.where(triangular_matrix == 0, '', triangular_matrix)
        assets_columns = ['','Strategy','S&P','VIX','Treasury \nrates', 'Gold', 'BTC']
        data = [[''] + assets_columns[2:]]
        data += triangular_matrix.tolist()
        for index in range(len(data[:-1])):
            data[index+1].insert(index, assets_columns[index+1])
        return data

    data = _clean_data()
    style = TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('LINEABOVE', (1,0), (-1,0), 0.5, pdf_util.colors['grey'], 'squared'), ('LINEBELOW', (-2,-1), (-1,-1), 0.5, pdf_util.colors['grey'], 'squared'),
                        ('LINEAFTER', (-1,0), (-1,0), 0.5, pdf_util.colors['grey'], 'squared'), ('LINEAFTER', (-1,2), (-1,-1), 0.5, pdf_util.colors['grey'], 'squared'),
                        ('LINEABOVE', (2,3), (2,3), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)), ('LINEABOVE', (3,4), (3,4), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)),
                        ('LINEABOVE', (4,5), (4,5), 0.5, pdf_util.colors['grey'], 'squared', (2,2,2)), ('LINEBELOW', (1,2), (1,2), 0.5, pdf_util.colors['grey'], 'squared'),
                        ('LINEBELOW', (2,3), (2,3), 0.5, pdf_util.colors['grey'], 'squared'), ('LINEBELOW', (3,4), (3,4), 0.5, pdf_util.colors['grey'], 'squared'),
                        ('BOX', (0,1), (-1,1), 1, pdf_util.colors['blue'], 'squared')])
    light_grey_background_coords = [[(1,0),(-1,0)], [(0,1),(0,1)], [(1,2),(1,2)], [(2,3),(2,3)], [(3,4),(3,4)], [(4,5),(4,5)]]
    style = pdf_util.table_background(style, light_grey_background_coords, 'light_grey')
    grey_line_before = [[(1,0),(1,0)], [(1,2),(1,2)], [(2,3),(2,3)], [(3,4),(3,4)], [(4,5),(4,5)]]
    style = pdf_util.table_line(style, 'before', grey_line_before, 'grey', 0.5)
    grey_line_after = [[(1,2),(1,2)], [(2,3),(2,3)], [(3,4),(3,4)], [(4,5),(4,5)]]
    style = pdf_util.table_line(style, 'after', grey_line_after, 'grey', 0.5, (2,2,2))
    pdf_util.add_table(data=data, row_heights=0.92*cm, col_widths=1.8*cm, style=style, x_coord=(pdf_util.PAGE_DIM['width']-10.8*cm)/2, y_coord=175.5)
    text = ''' This correlation matrix showcases the asset returns behaviour since January 2020. It highlights how these assets moved together or apart during this time. Each cell
    in the matrix is a coefficient, showing the correlation between two assets' returns. A number close to 1, means that the returns of the 2 assets usually move together. If it's around -1,
    they tend to move in opposite directions. If the number is near 0, there's not much connection. '''
    pdf_util.add_paragraph(text, 99)

    pdf.showPage()



strategy_data = PortfolioData(nowcasting_strat['weights'], benchmark['weights'], save_path='C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Reports/Ressources/Graphics')
returns_data = ReturnData(strategy_data)
risk_data = RiskData(strategy_data)


## updates graph
strategy_data.eco_cdt_plot()

returns_data.strat_vs_bench_perf()
returns_data.strat_vs_no_eom_rebalance_perf()
returns_data.returns_value_plot()
returns_data.returns_distrib_plot()

risk_data.std_risk_plot()
risk_data.dd_plot()
risk_data.beta_plot()
risk_data.std_distrib_plot()

z=strategy_data.active_weights()

#%%

## creates pdf file
pdf = canvas.Canvas(f'{factsheet_save_path}/{strategy_data.dates_meta()["period2"]} - Monthly factsheet - {strategy_data.dates_meta()["period"]}.pdf')
pdf_util = PdfMaker(pdf, strategy_data)


## creates page 1 (intro)
page_1(pdf, pdf_util, strategy_data)


## RETURNS
returns_part_1(pdf, pdf_util, returns_data)
returns_part_2(pdf, pdf_util, returns_data)
returns_part_3(pdf, pdf_util, returns_data)


## RISK
risk_part_1(pdf, pdf_util, risk_data)
risk_part_2(pdf, pdf_util, risk_data)
risk_part_3(pdf, pdf_util, risk_data)





strategy_data.dates_meta()





pdf.save()








