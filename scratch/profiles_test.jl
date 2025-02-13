using nuPGCM
using Printf
using JLD2
using PyPlot

include("../plots/derivatives.jl")

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

function main(γ)
    # dictionary of variables
    u_dict = Dict()
    v_dict = Dict()
    w_dict = Dict()
    b_dict = Dict()
    z_dict = Dict()

    # load 2D profiles
    # file = jldopen(@sprintf("../out/data/state1D_%0.5f.jld2", γ), "r")
    # file = jldopen(@sprintf("../out/data/state1D_old_b_%0.5f.jld2", γ), "r")
    # file = jldopen(@sprintf("../out/data/state1D_diff_%0.5f.jld2", γ), "r")
    file = jldopen(@sprintf("../out/data/state1D_diff_0.005_%0.5f.jld2", γ), "r")
    u_dict["1D"] = file["u"]
    v_dict["1D"] = file["v"]
    w_dict["1D"] = file["w"]
    b_dict["1D"] = file["b"]
    z_dict["1D"] = file["z"]
    close(file)

    # load 2D profiles
    # file = jldopen(@sprintf("../out/data/state2D_column_%0.5f.jld2", γ), "r")
    # file = jldopen(@sprintf("../out/data/state2D_old_b_column_%0.5f.jld2", γ), "r")
    # file = jldopen(@sprintf("../out/data/state2D_diff_column_%0.5f.jld2", γ), "r")
    file = jldopen(@sprintf("../out/data/state2D_diff_0.005_column_%0.5f.jld2", γ), "r")
    u_dict["2D"] = file["u"]
    v_dict["2D"] = file["v"]
    w_dict["2D"] = file["w"]
    b_dict["2D"] = file["b"]
    z_dict["2D"] = file["z"]
    close(file)

    # load 3D profiles
    fname = @sprintf("../out/data/state3D_column_%0.5f_%1.0e.jld2", γ, 1e-8)
    if isfile(fname)
        file = jldopen(fname, "r")
        u_dict["3D"] = file["u"]
        v_dict["3D"] = file["v"]
        w_dict["3D"] = file["w"]
        b_dict["3D"] = file["b"]
        z_dict["3D"] = file["z"]
        close(file)
    end

    # plot
    pc = 1/6
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 33pc/4*1.62), sharey=true)
    ax[1].set_title(L"\alpha = 1/"*@sprintf("%d", 1/γ))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    ax[4].set_xlabel(L"\partial_z b")
    ax[1].set_xlim(-1.5e-3, 5e-3)
    ax[2].set_xlim(-1e-2, 1.5e-1)
    ax[3].set_xlim(-1.5e-3, 5e-3)
    ax[4].set_xlim(0, 1.5)
    for a ∈ ax[1:3]
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", lw=0.5)
    end
    for a ∈ ax 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    end
    for dim ∈ ["1D", "2D"]
    # for dim ∈ ["1D", "2D", "3D"]
        mask = isnan.(u_dict[dim]) .== false
        u = u_dict[dim][mask]
        v = v_dict[dim][mask]
        w = w_dict[dim][mask]
        b = b_dict[dim][mask]
        z = z_dict[dim][mask]
        bz = differentiate(b, z)
        ax[1].plot(u,       z, label=dim)
        ax[2].plot(v,       z, label=dim)
        ax[3].plot(w,       z, label=dim)
        ax[4].plot(1 .+ bz, z, label=dim)
    end

    # # thermal wind
    # x = 0.5
    # f = 1
    # θ = π/4
    # h = 0.1
    # H = 1 - x^2
    # Hx = -2x 
    # Hxx = -2
    # println(Hx/(1 + γ*Hx^2))
    # println(2*γ*h*Hx*Hxx/(1 + γ*Hx^2)^2)
    # bx = @. -(Hx/(1 + γ*Hx^2) + 2*γ*h*Hx*Hxx/(1 + γ*Hx^2)^2) * exp(-(z_dict["2D"] + H)/h)
    # bζ = @. tan(θ)/(1 + γ*tan(θ)^2) * exp(-(z_dict["1D"] + H)/h)
    # v_tw1D = cumtrapz(bζ/f, z_dict["1D"])
    # v_tw2D = cumtrapz(bx/f, z_dict["1D"])
    # ax[2].plot(v_tw1D, z_dict["1D"], "C0--", lw=0.5, label="TW 1D")
    # ax[2].plot(v_tw2D, z_dict["1D"], "C1--", lw=0.5, label="TW 2D")

    ax[2].legend(loc=(-0.8, 0.5))
    # savefig(@sprintf("../out/images/profiles_%.5f.png", γ))
    # println(@sprintf("../out/images/profiles_%.5f.png", γ))
    # savefig(@sprintf("../out/images/profiles_old_b_%.5f.png", γ))
    # println(@sprintf("../out/images/profiles_old_b_%.5f.png", γ))
    # savefig(@sprintf("../out/images/profiles_diff_%.5f.png", γ))
    # println(@sprintf("../out/images/profiles_diff_%.5f.png", γ))
    savefig(@sprintf("../out/images/profiles_diff_0.005_%.5f.png", γ))
    println(@sprintf("../out/images/profiles_diff_0.005_%.5f.png", γ))
    plt.close()
end

main(1/4)
# main(1/8)