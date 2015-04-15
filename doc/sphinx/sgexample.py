from sphinx.directives.code import LiteralInclude
from docutils import nodes
import os
import uuid
import sys

PATH_TO_SHOGUN_META = "../../examples/example-generation/"
sys.path.insert(0, PATH_TO_SHOGUN_META)
import parse
import translate
import json

def setup(app):
    app.add_config_value('target_languages', None, True)

    # register functions called upon node-visiting
    app.add_node(sgexample,
            html=(visit_sgexample_node, depart_sgexample_node))
    app.add_node(tabpanel,
            html=(visit_tabpanel_node, depart_tabpanel_node))
    app.add_node(navtabs,
            html=(visit_navtabs_node, depart_navtabs_node))
    app.add_node(navtab,
            html=(visit_navtab_node, depart_navtab_node))
    app.add_node(fluid_tab_content,
            html=(visit_fluid_tab_content, depart_fluid_tab_content))

    app.add_directive('sgexample', ShogunExample)
    app.connect('builder-inited', setup_languages)

    return {'version': '0.1'}

class LocalContext(object):
    pass

context = LocalContext()

def setup_languages(app):
    context.target_languages = app.config.target_languages

class sgexample(nodes.Element):
    pass
class fluid_tab_content(nodes.Element):
    pass
class tabpanel(nodes.Element):
    pass
class navtabs(nodes.Element):
    pass
class navtab(nodes.Element):
    pass

def visit_tabpanel_node(self, node):
    self.body.append('<div role="tabpanel">')
def depart_tabpanel_node(self, node):
    self.body.append('</div>')

def visit_navtabs_node(self, node):
    self.body.append('<ul style="display:none" id="tabs-%s" class="nav nav-tabs" role="tablist">' % node.uid)
def depart_navtabs_node(self, node):
    self.body.append('</ul>')

def visit_navtab_node(self, node):
    cls = ""
    if node.index is 0:
        cls = 'class="active"'
    self.body.append('<li role="presentation" id="tab-%s" %s><a href="#%s" aria-controls="%s" role="tab" data-toggle="tab">' % (node.language, cls, node.language, node.language))
def depart_navtab_node(self, node):
    self.body.append('</a></li>')

def visit_fluid_tab_content(self, node):
    self.body.append('<div class="fluid-container tab-content">')
def depart_fluid_tab_content(self, node):
    self.body.append('</div>')

def visit_sgexample_node(self, node):
    cls = ""
    if node.index is 0:
        cls = 'active'
    self.body.append('<div role="tabpanel" class="tab-pane %s" id="%s">' % (cls, node.language))
def depart_sgexample_node(self, node):
    self.body.append('</div>')

class ShogunExample(LiteralInclude):
    def element_id(self, target, uid):
        return '%s-code-%s' % (target, uid)
    def resolve_path(self, name):
        return self.state.document.settings.env.relfn2path(name)
    def convert_example(self, fname, target, extension):
        source_path = self.resolve_path(fname)[1]
        target_path = "%s/targets/%s.json" % (PATH_TO_SHOGUN_META, target)
        destination_path = self.resolve_path(fname+'.'+extension)[1]

        with open(source_path, 'r') as source, \
             open(target_path, 'r') as translator, \
             open(destination_path,'w') as destination:
            ast = parse.parse(source.read(), source_path)
            language = json.load(translator)
            translated = translate.translate(ast, language)
            destination.write(translated)

    def run(self):

        section = self.arguments[0].split(':')[1]
        if section == 'begin':
            self.options['end-before'] = section
        elif section == 'end':
            self.options['start-after'] = section
        else:
            self.options['start-after'] = section
            self.options['end-before'] = section
        uid = str(uuid.uuid1())[:6]
        result = tabpanel()
        nvtbs = navtabs()
        nvtbs.uid = uid
        for i, (target, _) in enumerate(context.target_languages):
            nvtb = navtab()
            nvtb.language = self.element_id(target, uid)
            nvtb.index = i
            nvtbs += nvtb

        result += nvtbs

        # save original node
        fname = self.arguments[0].split(':')[0].strip()

        tbcntnt = fluid_tab_content()

        # create nodes with parsed listings
        for i, (target, extension) in enumerate(context.target_languages):
            self.convert_example(fname, target, extension)
            self.arguments[0] = fname+'.'+extension
            self.options['language'] = target
            # call base class, returns list
            include_container = sgexample()
            include_container.language = self.element_id(target, uid)
            include_container.index = i
            include_container += LiteralInclude.run(self)
            tbcntnt += include_container
        result += tbcntnt

        return [result]

def meta_convert(fname):
    pass
