struct Model
    arch::AbstractArchitecture
    mesh::Mesh
    params::Parameters
    inversion::InversionToolkit
    evolution::EvolutionToolkit
    state::State
end