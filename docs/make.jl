using Documenter
using DocumenterCitations
# using Literate

# const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
# const OUTPUT_DIR = joinpath(@__DIR__, "src/", "literated")

# withenv("JULIA_DEBUG" => "Literate") do
#     Literate.markdown(joinpath(EXAMPLES_DIR, "bowl_mixing.jl"), OUTPUT_DIR; 
#                       execute=true, 
#                       documenter=true)
# end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

model_formulation = [
    "The PG Equations" => "model_formulation/pg_equations.md"
    "Nondimensionalization" => "model_formulation/nondimensionalization.md"
    "Numerical Approach" => "model_formulation/numerical_approach.md"
]

# example_pages = [
#     "Mixing in a bowl" => "literated/bowl_mixing.md"
# ]

pages = [
    "Overview" => "index.md"
    "Model Formulation" => model_formulation
    # "Examples" => example_pages
    "References" => "references.md"
]

assets = String["assets/citations.css"]
mathengine = MathJax3(Dict(
    :tex => Dict(
        :inlineMath => [["\$","\$"], ["\\(","\\)"]],
        :tags => "ams",
        :packages => ["base", "ams", "autoload", "configmacros"],
        :macros => Dict(
            :pder => ["\\frac{\\partial #1}{\\partial #2}", 2],
            :prettyint => ["\\int_{#1}^#2 #3 \\; \\text{d}#4", 4],
            :nd => "\\tilde",
            :vec => "\\boldsymbol",
        )
    ),
))
format = Documenter.HTML(; collapselevel=1, 
                         assets, 
                         mathengine)

makedocs(; sitename="νPGCM",
         plugins=[bib],
         pages,
         format)

deploydocs(repo="github.com/hgpeterson/nuPGCM.git")