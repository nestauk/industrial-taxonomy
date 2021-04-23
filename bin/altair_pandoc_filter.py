#!/usr/bin/env python
"""Altair element format:

`altair_div` filter syntax:
```<div class=altair s3_path="test.json" static_path="png/bivariate_scatters.png" id="fig:pipeline2" width="500" height="20%">
can haz $ma^t_h$ again?
</div>
```

`altair_xml` filter syntax:
`<altair s3_path="test.json" static_path="png/bivariate_scatters.png" caption="$E = mc^2$" id="fig:pipeline3" width="300"/>`
"""
import string
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict

from pandocfilters import (
    toJSONFilter,
    Image,
    RawInline,
    Div,
    Span,
    Plain,
    Para,
    Str,
)

Pandoc = Callable[[Any], Dict[str, Any]]

script = """
<script type="text/javascript">
  var spec = "{s3_path}";
  vegaEmbed('#{div_id}', spec, {options}).then(function(result) {{
  }}).catch(console.error);
</script>""".format


def altair_img(key, value, format, meta):
    if (
        format == "html"
        # Capture Paragraph
        and key == "Para"
        # image of class "altair" as only child, i.e. a figure
        and value[0]["t"] == "Image"
        and len(value) == 1
        and "altair" in value[0]["c"][0][1]
    ):
        key, value = "Image", value[0]["c"]
        img_meta, img_caption, (img_content, _) = value

        reference = img_meta[0]
        classes = img_meta[1]
        properties = dict(img_meta[2])

        bucket = meta["bucket"]["c"]
        s3_key = Path(img_content).with_suffix(".json")
        s3_path = f"https://{bucket}.s3.amazonaws.com/{s3_key}"

        altair_div = vega_embed_div(s3_path, properties)

        # Replace Para with Div containing:
        # - vega-embed div (wrapped in Plain to obey AST)
        # - empty undisplayed image containing caption
        #   (wrapped in Para to generate a figure when pandoc-crossref runs)
        return Div(
            ("", [], list(properties.items())),
            [
                Plain([altair_div]),
                Para([dummy_img(reference, img_caption)]),
            ],
        )


def dummy_img(reference: str, caption: Pandoc) -> Image:
    """Return empty image with reference that will only display caption."""
    return Image(
        [
            reference,
            [],
            [
                ("onerror", "this.style.display='none'"),
            ],
        ],
        caption,
        [
            "data:,",
            "fig:",
        ],
    )


def vega_embed_div(s3_path: str, properties: dict) -> RawInline:
    """Generate `RawInline` vega-embed div & script."""

    div_id = "".join(random.sample(string.ascii_letters, 5))
    div = f'<div id="{div_id}"></div>'
    options = properties.copy()
    options.setdefault("actions", False)

    return RawInline(
        "html",
        div
        + script(
            div_id=div_id,
            s3_path=s3_path,
            options=json.dumps(options),
        )
    )


if __name__ == "__main__":
    toJSONFilter(altair_img)
