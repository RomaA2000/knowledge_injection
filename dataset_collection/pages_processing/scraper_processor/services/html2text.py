try:
    True
except NameError:
    setattr(__builtins__, "True", 1)
    setattr(__builtins__, "False", 0)


def has_key(x, y):
    if hasattr(x, "has_key"):
        return x.has_key(y)
    else:
        return y in x


try:
    import htmlentitydefs
    import HTMLParser
    import urlparse
except ImportError:  # Python3
    import html.entities as htmlentitydefs
    import html.parser as HTMLParser
    import urllib.parse as urlparse
try:  # Python3
    pass
except:
    pass

import codecs
import re
import sys

try:
    from textwrap import wrap
except:
    pass

# Use Unicode characters instead of their ascii psuedo-replacements
UNICODE_SNOB = 0

# Escape all special characters.  Output is less readable, but avoids corner case formatting issues.
ESCAPE_SNOB = 0

# Put the links after each paragraph instead of at the end.
LINKS_EACH_PARAGRAPH = 0

# Wrap long lines at position. 0 for no wrapping. (Requires Python 2.3.)
BODY_WIDTH = 78

# Don't show internal links (href="#local-anchor") -- corresponding link targets
# won't be visible in the plain text file anyway.
SKIP_INTERNAL_LINKS = True

# Use inline, rather than reference, formatting for images and links
INLINE_LINKS = True

# Number of pixels Google indents nested lists
GOOGLE_LIST_INDENT = 36

# Don't add markdown elements and output nicely for plain reading
NO_MARKDOWN = True

IGNORE_ANCHORS = True
# IGNORE_IMAGES = True
IGNORE_EMPHASIS = True

### Entity Nonsense ###


def name2cp(k):
    if k == "apos":
        return ord("'")
    if hasattr(htmlentitydefs, "name2codepoint"):  # requires Python 2.3
        return htmlentitydefs.name2codepoint[k]
    else:
        k = htmlentitydefs.entitydefs[k]
        if k.startswith("&#") and k.endswith(";"):
            return int(k[2:-1])  # not in latin-1
        return ord(codecs.latin_1_decode(k)[0])


unifiable = {
    "rsquo": "'",
    "lsquo": "'",
    "rdquo": '"',
    "ldquo": '"',
    "copy": "(C)",
    "mdash": "--",
    "nbsp": " ",
    "rarr": "->",
    "larr": "<-",
    "middot": "*",
    "ndash": "-",
    "oelig": "oe",
    "aelig": "ae",
    "agrave": "a",
    "aacute": "a",
    "acirc": "a",
    "atilde": "a",
    "auml": "a",
    "aring": "a",
    "egrave": "e",
    "eacute": "e",
    "ecirc": "e",
    "euml": "e",
    "igrave": "i",
    "iacute": "i",
    "icirc": "i",
    "iuml": "i",
    "ograve": "o",
    "oacute": "o",
    "ocirc": "o",
    "otilde": "o",
    "ouml": "o",
    "ugrave": "u",
    "uacute": "u",
    "ucirc": "u",
    "uuml": "u",
    "lrm": "",
    "rlm": "",
}

# All types of possible quotation marks - this is used to strip any blockquotes
# before we add our own quotes in, for plain text formatting
all_quotes = "\u0022\u0027\u00AB\u00BB\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F\u2039\u203A"

unifiable_n = {}

for k in unifiable.keys():
    unifiable_n[name2cp(k)] = unifiable[k]

### End Entity Nonsense ###


def onlywhite(line):
    """Return true if the line does only consist of whitespace characters."""
    for c in line:
        if c != " " and c != "  ":
            return c == " "
    return line


def hn(tag):
    if tag[0] == "h" and len(tag) == 2:
        try:
            n = int(tag[1])
            if n in range(1, 10):
                return n
        except ValueError:
            return 0


def dumb_property_dict(style):
    """returns a hash of css attributes"""
    return dict(
        [
            (x.strip(), y.strip())
            for x, y in [z.split(":", 1) for z in style.split(";") if ":" in z]
        ]
    )


def dumb_css_parser(data):
    """returns a hash of css selectors, each of which contains a hash of css attributes"""
    # remove @import sentences
    data += ";"
    importIndex = data.find("@import")
    while importIndex != -1:
        data = data[0:importIndex] + data[data.find(";", importIndex) + 1 :]
        importIndex = data.find("@import")

    # parse the css. reverted from dictionary compehension in order to support older pythons
    elements = [x.split("{") for x in data.split("}") if "{" in x.strip()]
    try:
        elements = dict([(a.strip(), dumb_property_dict(b)) for a, b in elements])
    except ValueError:
        elements = {}  # not that important

    return elements


def element_style(attrs, style_def, parent_style):
    """returns a hash of the 'final' style attributes of the element"""
    style = parent_style.copy()
    if "class" in attrs:
        for css_class in attrs["class"].split():
            css_style = style_def["." + css_class]
            style.update(css_style)
    if "style" in attrs:
        immediate_style = dumb_property_dict(attrs["style"])
        style.update(immediate_style)
    return style


def google_list_style(style):
    """finds out whether this is an ordered or unordered list"""
    if "list-style-type" in style:
        list_style = style["list-style-type"]
        if list_style in ["disc", "circle", "square", "none"]:
            return "ul"
    return "ol"


def google_has_height(style):
    """check if the style of the element has the 'height' attribute explicitly defined"""
    if "height" in style:
        return True
    return False


def google_text_emphasis(style):
    """return a list of all emphasis modifiers of the element"""
    emphasis = []
    if "text-decoration" in style:
        emphasis.append(style["text-decoration"])
    if "font-style" in style:
        emphasis.append(style["font-style"])
    if "font-weight" in style:
        emphasis.append(style["font-weight"])
    return emphasis


def google_fixed_width_font(style):
    """check if the css of the current element defines a fixed width font"""
    font_family = ""
    if "font-family" in style:
        font_family = style["font-family"]
    if "Courier New" == font_family or "Consolas" == font_family:
        return True
    return False


def list_numbering_start(attrs):
    """extract numbering from list element attributes"""
    if "start" in attrs:
        try:
            # print("attrs['start']",attrs['start'])
            return int(attrs["start"]) - 1
        except:
            return 0
    else:
        return 0


class HTML2Text(HTMLParser.HTMLParser):
    def __init__(self, out=None, baseurl="", ignore_image=True):
        HTMLParser.HTMLParser.__init__(self)

        # Config options
        self.unicode_snob = UNICODE_SNOB
        self.escape_snob = ESCAPE_SNOB
        self.links_each_paragraph = LINKS_EACH_PARAGRAPH
        self.body_width = BODY_WIDTH
        self.skip_internal_links = SKIP_INTERNAL_LINKS
        self.inline_links = INLINE_LINKS
        self.google_list_indent = GOOGLE_LIST_INDENT
        self.no_markdown = NO_MARKDOWN
        self.ignore_links = IGNORE_ANCHORS
        self.ignore_images = ignore_image
        self.ignore_emphasis = IGNORE_EMPHASIS
        self.google_doc = False
        self.ul_item_mark = "*"
        self.emphasis_mark = "_"
        self.strong_mark = "**"
        self.hr_mark = "* * *"
        self.blockquote_marks = ("> ", "")

        if out is None:
            self.out = self.outtextf
        else:
            self.out = out

        self.outtextlist = (
            []
        )  # empty list to store output characters before they are "joined"

        try:
            self.outtext = unicode()
        except NameError:  # Python3
            self.outtext = str()

        self.quiet = 0
        self.p_p = 0  # number of newline character to print before next output
        self.outcount = 0
        self.start = 1
        self.space = 0
        self.a = []
        self.astack = []
        self.maybe_automatic_link = None
        self.absolute_url_matcher = re.compile(r"^[a-zA-Z+]+://")
        self.acount = 0
        self.list = []
        self.blockquote = 0
        self.pre = 0
        self.startpre = 0
        self.code = False
        self.br_toggle = ""
        self.lastWasNL = 0
        self.lastWasList = False
        self.style = 0
        self.style_def = {}
        self.tag_stack = []
        self.emphasis = 0
        self.drop_white_space = 0
        self.inheader = False
        self.abbr_title = None  # current abbreviation definition
        self.abbr_data = None  # last inner HTML (for abbr being defined)
        self.abbr_list = {}  # stack of abbreviations to write later
        self.baseurl = baseurl
        self.last_tag_started = None  # holds the most recent tag we entered

        try:
            del unifiable_n[name2cp("nbsp")]
        except KeyError:
            pass
        unifiable["nbsp"] = "&nbsp_place_holder;"

    def normalise_options(self):
        """Configure options just before handle"""
        if self.no_markdown:
            # Configure for plain text output
            self.body_width = 0
            self.escape_snob = False
            self.ignore_links = True
            self.ignore_images = self.ignore_images  ##
            self.ignore_emphasis = True
            if self.unicode_snob:
                self.ul_item_mark = "\u2013"
                self.blockquote_marks = ("\u201C", "\u201D")
                self.hr_mark = "\u2014\u2014\u2014"
            else:
                self.ul_item_mark = "-"
                self.blockquote_marks = ('"', '"')
                self.hr_mark = "---"

    def feed(self, data):
        data = data.replace("</' + 'script>", "</ignore>")
        HTMLParser.HTMLParser.feed(self, data)

    def handle(self, data):
        self.normalise_options()
        self.feed(data)
        self.feed(" ")
        return self.post_process(self.close())

    def outtextf(self, s):
        self.outtextlist.append(s)
        if s:
            self.lastWasNL = s[-1] == "\n"

    def close(self):
        HTMLParser.HTMLParser.close(self)

        self.pbr()
        self.o("", 0, "end")

        self.outtext = self.outtext.join(self.outtextlist)
        if self.unicode_snob:
            nbsp = unichr(name2cp("nbsp"))
        else:
            nbsp = " "
        self.outtext = self.outtext.replace("&nbsp_place_holder;", nbsp)

        return self.outtext

    def handle_charref(self, c):
        self.o(self.charref(c), 1)

    def handle_entityref(self, c):
        self.o(self.entityref(c), 1)

    def handle_starttag(self, tag, attrs):
        self.handle_tag(tag, attrs, 1)

    def handle_endtag(self, tag):
        self.handle_tag(tag, None, 0)

    def previousIndex(self, attrs):
        """returns the index of certain set of attributes (of a link) in the
        self.a list

        If the set of attributes is not found, returns None
        """
        if not has_key(attrs, "href"):
            return None

        i = -1
        for a in self.a:
            i += 1
            match = 0

            if has_key(a, "href") and a["href"] == attrs["href"]:
                if has_key(a, "title") or has_key(attrs, "title"):
                    if (
                        has_key(a, "title")
                        and has_key(attrs, "title")
                        and a["title"] == attrs["title"]
                    ):
                        match = True
                else:
                    match = True

            if match:
                return i

    def drop_last(self, nLetters):
        if not self.quiet:
            self.outtext = self.outtext[:-nLetters]

    def handle_emphasis(self, start, tag_style, parent_style):
        """handles various text emphases"""
        tag_emphasis = google_text_emphasis(tag_style)
        parent_emphasis = google_text_emphasis(parent_style)

        # handle Google's text emphasis
        strikethrough = "line-through" in tag_emphasis and self.hide_strikethrough
        bold = "bold" in tag_emphasis and not "bold" in parent_emphasis
        italic = "italic" in tag_emphasis and not "italic" in parent_emphasis
        fixed = (
            google_fixed_width_font(tag_style)
            and not google_fixed_width_font(parent_style)
            and not self.pre
        )

        if start:
            # crossed-out text must be handled before other attributes
            # in order not to output qualifiers unnecessarily
            if bold or italic or fixed:
                self.emphasis += 1
            if strikethrough:
                self.quiet += 1
            if italic:
                self.o(self.emphasis_mark)
                self.drop_white_space += 1
            if bold:
                self.o(self.strong_mark)
                self.drop_white_space += 1
            if fixed:
                self.o("`")
                self.drop_white_space += 1
                self.code = True
        else:
            if bold or italic or fixed:
                # there must not be whitespace before closing emphasis mark
                self.emphasis -= 1
                self.space = 0
                self.outtext = self.outtext.rstrip()
            if fixed:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(1)
                    self.drop_white_space -= 1
                else:
                    self.o("`")
                self.code = False
            if bold:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(2)
                    self.drop_white_space -= 1
                else:
                    self.o(self.strong_mark)
            if italic:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(1)
                    self.drop_white_space -= 1
                else:
                    self.o(self.emphasis_mark)
            # space is only allowed after *all* emphasis marks
            if (bold or italic) and not self.emphasis:
                self.o(" ")
            if strikethrough:
                self.quiet -= 1

    def handle_tag(self, tag, attrs, start):
        # attrs = fixattrs(attrs)
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)
        if start:
            self.last_tag_started = tag

        if self.google_doc:
            # the attrs parameter is empty for a closing tag. in addition, we
            # need the attributes of the parent nodes in order to get a
            # complete style description for the current element. we assume
            # that google docs export well formed html.
            parent_style = {}
            if start:
                if self.tag_stack:
                    parent_style = self.tag_stack[-1][2]
                tag_style = element_style(attrs, self.style_def, parent_style)
                self.tag_stack.append((tag, attrs, tag_style))
            else:
                dummy, attrs, tag_style = self.tag_stack.pop()
                if self.tag_stack:
                    parent_style = self.tag_stack[-1][2]

        if hn(tag):
            self.p()
            if not self.no_markdown:
                if start:
                    self.inheader = True
                    # self.o(hn(tag)*"#" + ' ')
                else:
                    self.inheader = False
                    return  # prevent redundant emphasis marks on headers

        if tag in ["p", "div"]:
            if self.google_doc:
                if start and google_has_height(tag_style):
                    self.p()
                else:
                    self.soft_br()
            else:
                self.p()

        if tag == "br" and start:
            self.o("  \n")

        if tag == "hr" and start:
            self.p()
            self.o(self.hr_mark)
            self.p()

        if tag in ["head", "style", "script"]:
            if start:
                self.quiet += 1
            else:
                self.quiet -= 1

        if tag == "style":
            if start:
                self.style += 1
            else:
                self.style -= 1

        if tag == "body":
            self.quiet = 0  # sites like 9rules.com never close <head>

        if tag == "blockquote":
            if start:
                self.p()
                self.o(self.blockquote_marks[0], 0, 1)
                self.start = 1
                self.blockquote += 1
            else:
                if self.no_markdown:
                    # remove whitespace and extra quotes before adding our own quotes
                    self.rstrip_outtext(all_quotes)
                self.o(self.blockquote_marks[1], 0, 1)
                self.blockquote -= 1
                self.p()

        if tag in ["em", "i", "u"] and not self.ignore_emphasis:
            self.o(self.emphasis_mark)
        if tag in ["strong", "b"] and not self.ignore_emphasis:
            self.o(self.strong_mark)
        if tag in ["del", "strike", "s"] and not self.no_markdown:
            if start:
                self.o("<" + tag + ">")
            else:
                self.o("</" + tag + ">")

        if self.google_doc:
            if not self.inheader and not self.no_markdown:
                # handle some font attributes, but leave headers clean
                self.handle_emphasis(start, tag_style, parent_style)

        if tag in ["code", "tt"] and not self.pre:
            self.o("`")  # TODO: `` `this` ``
        if tag == "abbr":
            if start:
                self.abbr_title = None
                self.abbr_data = ""
                if has_key(attrs, "title"):
                    self.abbr_title = attrs["title"]
            else:
                if self.abbr_title != None:
                    self.abbr_list[self.abbr_data] = self.abbr_title
                    self.abbr_title = None
                self.abbr_data = ""

        if tag == "a" and not self.ignore_links:
            if start:
                if has_key(attrs, "href") and not (
                    self.skip_internal_links and attrs["href"].startswith("#")
                ):
                    self.astack.append(attrs)
                    self.maybe_automatic_link = attrs["href"]
                else:
                    self.astack.append(None)
            else:
                if self.astack:
                    a = self.astack.pop()
                    if self.maybe_automatic_link:
                        self.maybe_automatic_link = None
                    elif a:
                        if self.inline_links:
                            self.o("](" + escape_md(a["href"]) + ")")
                        else:
                            i = self.previousIndex(a)
                            if i is not None:
                                a = self.a[i]
                            else:
                                self.acount += 1
                                a["count"] = self.acount
                                a["outcount"] = self.outcount
                                self.a.append(a)
                            self.o("][" + str(a["count"]) + "]")

        if tag == "img" and start and not self.ignore_images:
            if has_key(attrs, "src"):
                attrs["href"] = attrs["src"]
                alt = attrs.get("alt", "")
                self.o("![" + escape_md(alt) + "]")

                if self.inline_links:
                    self.o("(" + escape_md(attrs["href"]) + ")")
                else:
                    i = self.previousIndex(attrs)
                    if i is not None:
                        attrs = self.a[i]
                    else:
                        self.acount += 1
                        attrs["count"] = self.acount
                        attrs["outcount"] = self.outcount
                        self.a.append(attrs)
                    self.o("[" + str(attrs["count"]) + "]")

        if tag == "dl" and start:
            self.p()
        if tag == "dt" and not start:
            self.pbr()
        if tag == "dd" and start:
            self.o("    ")
        if tag == "dd" and not start:
            self.pbr()

        if tag in ["ol", "ul"]:
            # Google Docs create sub lists as top level lists
            if (not self.list) and (not self.lastWasList):
                self.p()
            if start:
                if self.google_doc:
                    list_style = google_list_style(tag_style)
                else:
                    list_style = tag
                numbering_start = list_numbering_start(attrs)
                self.list.append({"name": list_style, "num": numbering_start})
            else:
                if self.list:
                    self.list.pop()
            self.lastWasList = True
        else:
            self.lastWasList = False

        if tag == "li":
            self.pbr()
            if start:
                if self.list:
                    li = self.list[-1]
                else:
                    li = {"name": "ul", "num": 0}
                if self.google_doc:
                    nest_count = self.google_nest_count(tag_style)
                else:
                    nest_count = len(self.list)
                self.o("  " * nest_count)  # TODO: line up <ol><li>s > 9 correctly.
                if li["name"] == "ul":
                    self.o(self.ul_item_mark + " ")
                elif li["name"] == "ol":
                    li["num"] += 1
                    self.o(str(li["num"]) + ". ")
                self.start = 1

        if tag in ["table", "tr"] and start:
            self.p()
        if tag == "td":
            self.pbr()

        if tag == "pre":
            if start:
                self.startpre = 1
                self.pre = 1
            else:
                self.pre = 0
            self.p()

    def pbr(self):
        if self.p_p == 0:
            self.p_p = 1

    def p(self):
        self.p_p = 2

    def soft_br(self):
        self.pbr()
        self.br_toggle = "  "

    def o(self, data, puredata=0, force=0):
        if self.abbr_data is not None:
            self.abbr_data += data

        if not self.quiet:
            if self.google_doc:
                # prevent white space immediately after 'begin emphasis' marks ('**' and '_')
                lstripped_data = data.lstrip()
                if self.drop_white_space and not (self.pre or self.code):
                    data = lstripped_data
                if lstripped_data != "":
                    self.drop_white_space = 0

            if puredata and not self.pre:
                data = re.sub("\s+", " ", data)
                if data and data[0] == " ":
                    self.space = 1
                    data = data[1:]
            if not data and not force:
                return

            if self.startpre:
                # self.out(" :") #TODO: not output when already one there
                if not data.startswith("\n"):  # <pre>stuff...
                    data = "\n" + data

            if puredata and self.last_tag_started == "blockquote" and self.no_markdown:
                data = data.lstrip(" \t\n\r" + all_quotes)

            bq = ""
            if not self.no_markdown:
                bq = ">" * self.blockquote
                if not (force and data and data[0] == ">") and self.blockquote:
                    bq += " "

            if self.pre:
                if not self.list:
                    bq += "    "
                # else: list content is already partially indented
                for i in range(len(self.list)):
                    bq += "    "
                data = data.replace("\n", "\n" + bq)

            if self.startpre:
                self.startpre = 0
                if self.list:
                    data = data.lstrip("\n")  # use existing initial indentation

            if self.start:
                self.space = 0
                self.p_p = 0
                self.start = 0

            if force == "end":
                # It's the end.
                self.p_p = 0
                self.out("\n")
                self.space = 0

            if self.p_p:
                self.out((self.br_toggle + "\n" + bq) * self.p_p)
                self.space = 0
                self.br_toggle = ""

            if self.space:
                if not self.lastWasNL:
                    self.out(" ")
                self.space = 0

            if self.a and (
                (self.p_p == 2 and self.links_each_paragraph) or force == "end"
            ):
                if force == "end":
                    self.out("\n")

                newa = []
                for link in self.a:
                    if self.outcount > link["outcount"]:
                        self.out(
                            "   ["
                            + str(link["count"])
                            + "]: "
                            + urlparse.urljoin(self.baseurl, link["href"])
                        )
                        if has_key(link, "title"):
                            self.out(" (" + link["title"] + ")")
                        self.out("\n")
                    else:
                        newa.append(link)

                if self.a != newa:
                    self.out("\n")  # Don't need an extra line when nothing was done.

                self.a = newa

            if self.abbr_list and force == "end":
                for abbr, definition in self.abbr_list.items():
                    self.out("  *[" + abbr + "]: " + definition + "\n")

            self.p_p = 0
            self.out(data)
            self.outcount += 1

    def handle_data(self, data):
        if r"\/script>" in data:
            self.quiet -= 1

        if self.style:
            self.style_def.update(dumb_css_parser(data))

        if not self.maybe_automatic_link is None:
            href = self.maybe_automatic_link
            if href == data and self.absolute_url_matcher.match(href):
                self.o("<" + data + ">")
                return
            else:
                self.o("[")
                self.maybe_automatic_link = None

        if not self.code and not self.pre and not self.no_markdown:
            data = escape_md_section(data, snob=self.escape_snob)
        self.o(data, 1)

    def unknown_decl(self, data):
        pass

    def charref(self, name):
        if name[0] in ["x", "X"]:
            c = int(name[1:], 16)
        else:
            c = int(name)

        if not self.unicode_snob and c in unifiable_n.keys():
            return unifiable_n[c]
        else:
            try:
                return unichr(c)
            except NameError:  # Python3
                return chr(c)

    def entityref(self, c):
        if not self.unicode_snob and c in unifiable.keys():
            return unifiable[c]
        else:
            try:
                name2cp(c)
            except KeyError:
                if self.no_markdown:
                    # let original ampersand and character through
                    return "&" + c
                else:
                    return "&" + c + ";"
            else:
                try:
                    return unichr(name2cp(c))
                except NameError:  # Python3
                    return chr(name2cp(c))

    def replaceEntities(self, s):
        s = s.group(1)
        if s[0] == "#":
            return self.charref(s[1:])
        else:
            return self.entityref(s)

    r_unescape = re.compile(r"&(#?[xX]?(?:[0-9a-fA-F]+|\w{1,8}));")

    def unescape(self, s):
        return self.r_unescape.sub(self.replaceEntities, s)

    def google_nest_count(self, style):
        """calculate the nesting count of google doc lists"""
        nest_count = 0
        if "margin-left" in style:
            nest_count = int(style["margin-left"][:-2]) / self.google_list_indent
        return nest_count

    def post_process(self, text):
        if self.no_markdown:
            # Tidy up for plain text response
            text = remove_multi_blank_lines(text)
        else:
            # Wrapping does not work with plain text yet, as the criteria in skipwrap
            # depends on markdown formatting and syntax
            text = self.optwrap(text)
        return text

    def optwrap(self, text):
        """Wrap all paragraphs in the provided text."""
        if not self.body_width:
            return text

        assert wrap, "Requires Python 2.3."
        result = ""
        newlines = 0
        for para in text.split("\n"):
            if len(para) > 0:
                if not skipwrap(para):
                    result += "\n".join(wrap(para, self.body_width))
                    if para.endswith("  "):
                        result += "  \n"
                        newlines = 1
                    else:
                        result += "\n\n"
                        newlines = 2
                else:
                    if not onlywhite(para):
                        result += para + "\n"
                        newlines = 1
            else:
                if newlines < 2:
                    result += "\n"
                    newlines += 1
        return result

    def rstrip_outtext(self, additional_chars):
        """Remove whitespace at the end of the outtext"""
        if self.outtextlist:
            self.outtextlist[-1] = self.outtextlist[-1].rstrip(
                " \r\t\n" + additional_chars
            )


multi_blank_line_matcher = re.compile(r"([ \t]*\n){3,}")
ordered_list_matcher = re.compile(r"\d+\.\s")
unordered_list_matcher = re.compile(r"[-\*\+]\s")
md_chars_matcher = re.compile(r"([\\\[\]\(\)])")
md_chars_matcher_all = re.compile(r"([`\*_{}\[\]\(\)#!])")
md_dot_matcher = re.compile(
    r"""
    ^             # start of line
    (\s*\d+)      # optional whitespace and a number
    (\.)          # dot
    (?=\s)        # lookahead assert whitespace
    """,
    re.MULTILINE | re.VERBOSE,
)
md_plus_matcher = re.compile(
    r"""
    ^
    (\s*)
    (\+)
    (?=\s)
    """,
    flags=re.MULTILINE | re.VERBOSE,
)
md_dash_matcher = re.compile(
    r"""
    ^
    (\s*)
    (-)
    (?=\s|\-)     # followed by whitespace (bullet list, or spaced out hr)
                  # or another dash (header or hr)
    """,
    flags=re.MULTILINE | re.VERBOSE,
)
slash_chars = r"\`*_{}[]()#+-.!"
md_backslash_matcher = re.compile(
    r"""
    (\\)          # match one slash
    (?=[%s])      # followed by a char that requires escaping
    """
    % re.escape(slash_chars),
    flags=re.VERBOSE,
)


def skipwrap(para):
    # If the text begins with four spaces or one tab, it's a code block; don't wrap
    if para[0:4] == "    " or para[0] == "\t":
        return True
    # If the text begins with only two "--", possibly preceded by whitespace, that's
    # an emdash; so wrap.
    stripped = para.lstrip()
    if stripped[0:2] == "--" and len(stripped) > 2 and stripped[2] != "-":
        return False
    # I'm not sure what this is for; I thought it was to detect lists, but there's
    # a <br>-inside-<span> case in one of the tests that also depends upon it.
    if stripped[0:1] == "-" or stripped[0:1] == "*":
        return True
    # If the text begins with a single -, *, or +, followed by a space, or an integer,
    # followed by a ., followed by a space (in either case optionally preceeded by
    # whitespace), it's a list; don't wrap.
    if ordered_list_matcher.match(stripped) or unordered_list_matcher.match(stripped):
        return True
    return False


def wrapwrite(text):
    text = text.encode("utf-8")
    try:  # Python3
        sys.stdout.buffer.write(text)
    except AttributeError:
        sys.stdout.write(text)


def html2text(html, baseurl=""):
    h = HTML2Text(baseurl=baseurl)
    return h.handle(html)


def html2text_with_images(html, baseurl=""):
    h = HTML2Text(baseurl=baseurl, ignore_image=False)
    return h.handle(html)


def unescape(s, unicode_snob=False):
    h = HTML2Text()
    h.unicode_snob = unicode_snob
    return h.unescape(s)


def escape_md(text):
    """Escapes markdown-sensitive characters within other markdown constructs."""
    return md_chars_matcher.sub(r"\\\1", text)


def escape_md_section(text, snob=False):
    """Escapes markdown-sensitive characters across whole document sections."""
    text = md_backslash_matcher.sub(r"\\\1", text)
    if snob:
        text = md_chars_matcher_all.sub(r"\\\1", text)
    text = md_dot_matcher.sub(r"\1\\\2", text)
    text = md_plus_matcher.sub(r"\1\\\2", text)
    text = md_dash_matcher.sub(r"\1\\\2", text)
    return text


def remove_multi_blank_lines(text):
    """Ensure there can only be one blank line between text"""
    return multi_blank_line_matcher.sub("\n\n", text)
