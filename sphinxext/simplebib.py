import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from pybtex.database import parse_file
from sphinx.domains import Domain
from sphinx.util import logging

logger = logging.getLogger(__name__)


class cite_ref(nodes.Inline, nodes.Element):
    """Placeholder for one in-text citation; resolved to a numbered link."""


class bibliography_node(nodes.General, nodes.Element):
    """Placeholder for a page's reference list; resolved to a numbered list."""


def _cite_role(name, rawtext, text, lineno, inliner, options=None, content=None):  # noqa: ARG001
    result = []
    for i, key in enumerate(k.strip() for k in text.split(",") if k.strip()):
        if i:
            result.append(nodes.Text(", "))
        node = cite_ref("", key=key)
        result.append(node)
    return result, []


class CiteDomain(Domain):
    name = "cite"
    label = "simplebib"
    roles = {"t": _cite_role, "p": _cite_role}

    def get_objects(self):
        return []

    def resolve_xref(self, *args):  # noqa: ARG002
        return None


class BibliographyDirective(Directive):
    has_content = False
    # Accept and ignore sphinxcontrib-bibtex's options so existing pages parse.
    option_spec = {"filter": directives.unchanged, "keyprefix": directives.unchanged}

    def run(self):
        return [bibliography_node("")]


def _load_bib(app):
    entries = {}
    for name in app.config.simplebib_bibfiles:
        path = app.env.relfn2path(name, app.config.master_doc)[1]
        for key, entry in parse_file(path).entries.items():
            entries[key.lower()] = entry
    return entries


def _persons(entry):
    return entry.persons.get("author") or entry.persons.get("editor") or []


def _clean(value):
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


def _format_person(person):
    last = _clean(" ".join(person.prelast_names + person.last_names))
    given = [_clean(n) for n in person.first_names + person.middle_names]
    initials = " ".join(f"{n[0]}." for n in given if n)
    return f"{last}, {initials}".rstrip(", ").strip()


def _format_authors(persons):
    names = [_format_person(p) for p in persons]
    if len(names) <= 1:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + ", & " + names[-1]


def _format_entry(entry):
    fields = entry.fields
    authors = _format_authors(_persons(entry))
    year = fields.get("year", "n.d.")
    title = _clean(fields.get("title", ""))

    head = " ".join(part for part in [authors, f"({year})."] if part)
    text = f"{head} {title}." if title else head

    venue = _clean(
        fields.get("journal")
        or fields.get("booktitle")
        or fields.get("publisher")
        or fields.get("school")
        or fields.get("institution")
        or fields.get("howpublished")
        or ""
    )
    if venue:
        detail = venue
        if fields.get("volume"):
            detail += f", {fields['volume']}"
            if fields.get("number"):
                detail += f"({fields['number']})"
        if fields.get("pages"):
            detail += f", {_clean(fields['pages']).replace('--', '–')}"
        text += f" {detail}."
    return text


def _anchor(key):
    return nodes.make_id(f"ref-{key}")


def _resolve_page(app, doctree, docname):  # noqa: ARG001
    bib = app.env.simplebib_data
    numbers = {}  # key -> citation number (first-seen order)
    backrefs = {}  # key -> [ids of the in-text markers]

    for node in list(doctree.findall(cite_ref)):
        key = node["key"]
        if key not in bib:
            logger.warning(f"unknown citation key {key!r}", location=node, type="simplebib")
            node.replace_self(nodes.Text("[?]"))
            continue
        number = numbers.setdefault(key, len(numbers) + 1)
        ref_id = nodes.make_id(f"cite-{key}-{len(backrefs.get(key, []))}")
        backrefs.setdefault(key, []).append(ref_id)

        marker = nodes.reference("", f"[{number}]", internal=True, refid=_anchor(key))
        marker["ids"].append(ref_id)
        marker["classes"].append("simplebib-cite")
        node.replace_self(marker)

    for node in list(doctree.findall(bibliography_node)):
        if not numbers:
            node.replace_self([])
            continue
        listing = nodes.container(classes=["simplebib-references"])
        for key in sorted(numbers, key=numbers.get):
            para = nodes.paragraph(ids=[_anchor(key)], classes=["simplebib-reference"])
            para += nodes.Text(f"[{numbers[key]}] {_format_entry(bib[key])}")
            for ref_id in backrefs[key]:
                para += nodes.Text(" ")
                back = nodes.reference("", "↩", internal=True, refid=ref_id)
                back["classes"].append("simplebib-backref")
                para += back
            listing += para
        node.replace_self(listing)


def _noop(self, node):  # noqa: ARG001
    raise nodes.SkipNode


def _init_bib(app):
    app.env.simplebib_data = _load_bib(app)


def setup(app):
    app.add_config_value("simplebib_bibfiles", ["references.bib"], "env")
    app.add_domain(CiteDomain)
    app.add_directive("bibliography", BibliographyDirective)
    # These nodes are always replaced at doctree-resolved; the skip visitors
    # are a safety net for any builder if one ever survives.
    skip = (_noop, None)
    for node in (cite_ref, bibliography_node):
        app.add_node(node, html=skip, latex=skip, text=skip, man=skip, texinfo=skip)
    app.connect("builder-inited", _init_bib)
    app.connect("doctree-resolved", _resolve_page)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
