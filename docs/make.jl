using nuPGCM
using Documenter
using DocumenterCitations
using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/", "literated")

# withenv("JULIA_DEBUG" => "Literate") do
    Literate.markdown(joinpath(EXAMPLES_DIR, "bowl_mixing.jl"), OUTPUT_DIR; execute=true)
# end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

model_formulation = [
    "The PG Equations" => "pg_equations.md"
    "Nondimensionalization" => "nondimensionalization.md"
    "Numerical Approach" => "numerical_approach.md"
]

example_pages = [
    "Mixing in a bowl" => "literated/bowl_mixing.md"
]

pages = [
    "Overview" => "index.md"
    "Model Formulation" => model_formulation
    "Examples" => example_pages
]

assets = String["assets/citations.css"]
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