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
    # `cite:t` names the authors in the sentence; `cite:p` is the bare marker.
    textual = name.rpartition(":")[2] == "t"
    result = []
    for i, key in enumerate(k.strip() for k in text.split(",") if k.strip()):
        if i:
            result.append(nodes.Text(", "))
        node = cite_ref("", key=key, textual=textual)
        result.append(node)
    return result, []


class CiteDomain(Domain):
    name = "cite"
    label = "simplebib"
    roles = {"t": _cite_role, "p": _cite_role}

    def get_objects(self):
        return []

    # The domain stores no data, so there is nothing to merge. Sphinx's default raises, which blocks
    # parallel reads.
    def merge_domaindata(self, docnames, otherdata):
        pass

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):  # noqa: ARG002
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


def _last_name(person):
    return _clean(" ".join(person.prelast_names + person.last_names))


def _format_person(person):
    given = [_clean(n) for n in person.first_names + person.middle_names]
    initials = " ".join(f"{n[0]}." for n in given if n)
    return f"{_last_name(person)}, {initials}".rstrip(", ").strip()


def _format_authors(persons):
    names = [_format_person(p) for p in persons]
    if len(names) <= 1:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + ", & " + names[-1]


def _short_authors(persons):
    names = [_last_name(p) for p in persons]
    if len(names) > 2:
        return f"{names[0]} et al."
    return " and ".join(names)


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
                detail += f"({_format_range(fields['number'])})"
        if fields.get("pages"):
            detail += f", {_format_range(fields['pages'])}"
        text += f" {detail}."
    return text


def _format_range(value):
    return _clean(value).replace("--", "\N{EN DASH}")


def _doi_link(entry):
    doi = _clean(entry.fields.get("doi", ""))
    # Entries copied from publisher pages often carry the resolver URL rather than the bare DOI.
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    if not doi:
        return None
    return nodes.reference("", f"doi:{doi}", refuri=f"https://doi.org/{doi}")


def _anchor(key):
    return nodes.make_id(f"ref-{key}")


def _resolve_page(app, doctree, docname):
    bib = app.env.simplebib_data
    numbers = {}  # key -> citation number (first-seen order)
    backrefs = {}  # key -> [ids of the in-text markers]

    for node in list(doctree.findall(cite_ref)):
        # BibTeX keys are case-insensitive, and _load_bib stores them lowercased.
        key = node["key"].lower()
        if key not in bib:
            logger.warning(f"unknown citation key {node['key']!r}", location=node, type="simplebib")
            node.replace_self(nodes.Text("[?]"))
            continue
        number = numbers.setdefault(key, len(numbers) + 1)
        ref_id = nodes.make_id(f"cite-{key}-{len(backrefs.get(key, []))}")
        backrefs.setdefault(key, []).append(ref_id)

        marker = nodes.reference("", f"[{number}]", internal=True, refid=_anchor(key))
        marker["ids"].append(ref_id)
        marker["classes"].append("simplebib-cite")

        authors = _short_authors(_persons(bib[key])) if node["textual"] else ""
        node.replace_self([nodes.Text(f"{authors} "), marker] if authors else marker)

    bibliographies = list(doctree.findall(bibliography_node))
    if numbers and not bibliographies:
        logger.warning(
            f"{docname} cites {len(numbers)} reference(s) but has no bibliography, so its "
            "citation markers link nowhere",
            location=docname,
            type="simplebib",
        )

    for node in bibliographies:
        if not numbers:
            node.replace_self([])
            continue
        listing = nodes.container(classes=["simplebib-references"])
        for key, number in numbers.items():
            para = nodes.paragraph(ids=[_anchor(key)], classes=["simplebib-reference"])
            para += nodes.Text(f"[{number}] {_format_entry(bib[key])}")
            doi_link = _doi_link(bib[key])
            if doi_link is not None:
                para += nodes.Text(" ")
                para += doi_link
            for ref_id in backrefs[key]:
                para += nodes.Text(" ")
                back = nodes.reference("", "\N{LEFTWARDS ARROW WITH HOOK}", internal=True, refid=ref_id)
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
    # These nodes are always replaced at doctree-resolved; the skip visitors are a safety net for any
    # builder if one ever survives.
    skip = (_noop, None)
    for node in (cite_ref, bibliography_node):
        app.add_node(node, html=skip, latex=skip, text=skip, man=skip, texinfo=skip)
    app.connect("builder-inited", _init_bib)
    app.connect("doctree-resolved", _resolve_page)
    # Sphinx re-reads every page when env_version changes, so bump it whenever the attributes stored on
    # cite_ref or bibliography_node change. Pages cached under the old shape would otherwise reach
    # _resolve_page without them.
    return {
        "version": "0.1",
        "env_version": 1,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
