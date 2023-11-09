import os

def import_file_as_str(file_path):
    with open(file_path, 'r') as file:
        return file.read()
def save_str_as_file(file_path, str_to_save):
    with open(file_path, 'w') as file:
        file.write(str_to_save)

def fix_xticklabels(str_total):
    # identify the xticklabels
    str_xticklabels_org = str_total.split('xticklabels={')[1].split('y grid style')[0]
    # print(str_xticklabels_org)
    str_xticklabels_new = str_xticklabels_org.replace('\n', r'\\') # insert latex linebreaks
    # print(str_xticklabels_new)
    str_xticklabels_new = str_xticklabels_new.replace(r'},\\  \\',  '},\n{') # insert brackets 1
    str_xticklabels_new = str_xticklabels_new.replace(r'\\  \\',  '},\n{') # insert brackets 2
    str_xticklabels_new = str_xticklabels_new.replace(r'\\  {\\',  '') # fix beginning
    str_xticklabels_new = str_xticklabels_new.replace(r'\\},\\',  '}') # fix end
    str_xticklabels_new = str_xticklabels_new.replace(r',}',  '}') # fix end
    
    # add style
    xtick_label_style = 'xticklabel style={align=center},'
    str_xticklabels_new = '\n  {\n' + str_xticklabels_new + '\n  },\n'+ xtick_label_style
    
    str_total_new = str_total.replace(str_xticklabels_org, str_xticklabels_new, 1)
    str_total_new = str_total_new.replace(str_xticklabels_org, '},\n', 1)
    # print(str_total_new)
    return str_total_new

def add_variables(str_total, width = 0.8, heigth = 0.48, anchorx = 1.10):
    # initialize vairaibles
    to_add = '  \n  \\newcommand{\widthplot}{'+str(width)+'}\n  \\newcommand{\heightplot}{'+str(heigth)+'}\n  \\newcommand{\\anchorx}{'+str(anchorx)+'}\n'
    str_total_new = str_total.replace('begin{tikzpicture}', 'begin{tikzpicture}' + to_add, 1)
    
    # apply variables
    to_add = '  \n  width=\widthplot\\textwidth,\n  height=\heightplot\\textwidth,'
    str_total_new = str_total_new.replace('begin{axis}[', 'begin{axis}[' + to_add, 2)
    return str_total_new
    
def fix_anchor(str_total):
    # identify the text to replace
    # anchor 1
    to_replace = str_total.split('legend style={')[1].split('\n}')[0]
    to_replace_2 = to_replace.split('anchor=')[1].split('},')[0]
    to_replace_3 = to_replace.split('at=')[1].split('},')[0]
    replaced = to_replace.replace(to_replace_2, 'west', 1)
    replaced = replaced.replace(to_replace_3, '{(\\anchorx,0.00)', 1)
    str_total_new = str_total.replace(to_replace, replaced, 1)
    to_replace = 'south west'
    to_insert = ',\n  text width=2.25cm'
    str_total_new = str_total_new.replace(to_replace, to_replace + to_insert, 1)
    str_total_new
    # amchor 2
    to_insert = '\nlegend cell align={left},\nlegend style={\n  fill opacity=0.8,\n  draw opacity=1,\n  text opacity=1,\n  at={(\\anchorx,1.00)},\n  anchor=north west,\n  text width=2.25cm},'
    to_replace = 'height=\heightplot\\textwidth,\naxis y line=right,'
    str_total_new = str_total_new.replace(to_replace, to_replace + to_insert, 1)
    return str_total_new
    
def change_scale(str_total, scale_value=1.0):
    # identify the text to replace
    to_replace = 'scale=0.5'
    str_total_new = str_total.replace(to_replace, 'scale='+str(scale_value))
    return str_total_new

def adjust_offset_bar_label(str_total, offset_value=(0,0)):
    # identify the text to replace
    # to_replace = 'scale=0.5'
    to_replace = '++(0pt,3pt)'
    replace_with = f'++({offset_value[0]}pt,{offset_value[1]}pt)'
    str_total_new = str_total.replace(to_replace, replace_with)
    return str_total_new

def add_annotation(str_total, input_backbone_name, input_device_name):
    # identify the text to replace
    to_replace = 'height=\heightplot\\textwidth,\nlegend cell align={left},'
    to_insert = f'extra x ticks={{0}},\nextra x tick labels={{Backbone: {input_backbone_name}\\\\Device: {input_device_name}}},\nextra x tick style={{tickwidth=0pt}},\n'
    print(str_total.find(to_replace))
    str_total_new = str_total.replace(to_replace, to_replace + to_insert, 1)
    return str_total_new

def add_100_line(str_total):
    to_replace = '\end{axis}\n\n\end{tikzpicture}'
    to_insert = '\n\\draw[black, dashed] (axis cs:-1,100) -- (axis cs:5,100);'
    str_total_new = str_total.replace(to_replace, to_insert + to_replace, 1)
    return str_total_new
if __name__ == '__main__':
    print('Start')
    
    filepath = os.path.join(os.path.dirname(__file__), 'slideactivrn18j23.tex')
    str_total = import_file_as_str(filepath)
    # print(str_total[:-100])
    
    # str_total = fix_xticklabels(str_total)
    # str_total = add_variables(str_total)
    # str_total = fix_anchor(str_total)
    str_total = change_scale(str_total, 0.6)
    # str_total = adjust_offset_bar_label(str_total, (0,-2))    
    str_total = add_100_line(str_total)
    
    print(str_total)
    # str_total = add_annotation(str_total, 'ResNet-50', 'CPU')
    filepath = filepath.replace('.tex', '.tex')
    save_str_as_file(filepath, str_total)


# xticklabels={
#  {\num{0,1}\%},{\#1000\\ MW $\approx \num{0,6}$\%},{1\%},{10\%},{100\%}
#  },