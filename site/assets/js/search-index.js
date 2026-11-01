---
layout: null
---
{%- assign docs = site.pages | where_exp: "p", "p.url contains '/docs/'" -%}
{%- assign extra = site.pages | where_exp: "p", "p.search_include" -%}
window.HC_SEARCH_INDEX = [
{%- for p in docs -%}
{%- if p.title and p.url != '/docs/' and p.redirect_to == nil -%}
{"t":{{ p.title | jsonify }},"u":{{ p.url | relative_url | jsonify }},"c":{{ p.content | markdownify | strip_html | normalize_whitespace | truncate: 6000, "" | jsonify }}},
{%- endif -%}
{%- endfor -%}
{%- for p in extra -%}
{"t":{{ p.title | jsonify }},"u":{{ p.url | relative_url | jsonify }},"c":{{ p.content | markdownify | strip_html | normalize_whitespace | truncate: 6000, "" | jsonify }}},
{%- endfor -%}
null].filter(Boolean);
