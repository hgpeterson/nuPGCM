using nuPGCM
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

pages = [
    "Overview" => "index.md"
    "PG Equations" => "pg_equations.md"
    "Nondimensionalization" => "nondimensionalization.md"
]

assets = String["assets/citations.css"]  # this gets deleted??
mathengine = Documenter.KaTeX(
                Dict(:delimiters => [
                         Dict(:left => raw"$",   :right => raw"$",   display => false),
                         Dict(:left => raw"$$",  :right => raw"$$",  display => true),
                         Dict(:left => raw"\[",  :right => raw"\]",  display => true),
                     ],
                     :macros => Dict(
                                "\\pder" => "\\frac{\\partial #1}{\\partial #2}",
                                "\\prettyint" => "\\int_{#1}^#2 #3 \\; \\text{d}#4",
                                "\\nd" => "\\tilde",
                                "\\vec" => "\\bm",
                                ),
                    )
                )
format = Documenter.HTML(; assets, mathengine)

makedocs(; sitename = "νPGCM",
           plugins=[bib],
           pages,
           format)