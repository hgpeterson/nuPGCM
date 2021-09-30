using PyPlot, LinearAlgebra
plt.style.use("../plots.mplstyle")
close("all")
pygui(false)

"""
    stages, c, A_ex, A_im = RKTable(order) 

Returns number of `stages`, fractional time steps `c`, explicity coefficients `A_ex`,
and implicit coefficients `A_im` for Runge-Kutta scheme of order `order`. Implemented
options:

    order | description    
    -------------------
    111     1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]
    222     2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]
    443     3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8]
"""
function RKTable(order::String)
    if order == "111"
        # number of stages
        stages = 1

        # time steps between stages
        c = [0., 1]

        # explicit coefficients
        A_ex = [  0.    0
                  1     0]

        # implicit coefficients
        A_im = [0.  0;
                0   1]

    elseif order == "222"
        # number of stages
        stages = 2

        # useful variables
        γ = (2 - sqrt(2)) / 2
        δ = 1 - 1 / γ / 2

        # time steps between stages
        c = [0, γ, 1]

        # explicit coefficients
        A_ex = [0  0  0;
                γ  0  0;
                δ 1-δ 0]

        # implicit coefficients
        A_im = [0  0  0;
                0  γ  0;
                0 1-γ γ]
    elseif order == "443"
        # number of stages
        stages = 4

        # time steps between stages
        c = [0., 1/2, 2/3, 1/2, 1]

        # explicit coefficients
        A_ex = [  0.    0    0    0  0;
                 1/2    0    0    0  0;
                11/18  1/18  0    0  0;
                 5/6  -5/6  1/2   0  0;
                 1/4   7/4  3/4 -7/4 0]

        # implicit coefficients
        A_im = [0.  0    0   0   0 ;
                0  1/2   0   0   0 ;
                0  1/6  1/2  0   0 ;
                0 -1/2  1/2 1/2  0 ;
                0  3/2 -3/2 1/2 1/2]
    else
        error("Order ", order, "not implemented.")
    end

    return stages, c, A_ex, A_im
end

function RKStep(k, u0, stages, c, A_ex, A_im, f, g)
    # explict and implicit contrbutions at each stage 
    K_ex = zeros(Complex, stages+1)
    K_im = zeros(Complex, stages)
        
    # start with explicit contribution right now
    K_ex[1] = f(u0)

    # solve for each stage
    α = g(u0)/u0
    for i=1:stages
        LHS = 1 - A_im[i+1, i+1]*k*α
        RHS = u0 + k*A_ex[i+1, i]*K_ex[i]
        for j=1:i-1
            RHS += k*(A_im[i+1, j+1]*K_im[j] + A_ex[i+1, j]*K_ex[j])
        end
        ui = RHS/LHS
        K_im[i] = g(ui)
        K_ex[i+1] = f(ui)
    end    

    # sum up
    un = u0
    for j=1:stages
        un += k*(A_im[end, j+1]*K_im[j] + A_ex[end, j]*K_ex[j])
    end
    un += k*A_ex[end, stages+1]*K_ex[stages+1]

    return un
end

function run()
    α = -1
    β = 1
    function f(u)
        return im*β*u
    end
    function g(u)
        return α*u
    end

    k = 1e-4
    tFinal = 5
    t = 0:k:tFinal
    nSteps = size(t, 1)

    # stages, c, A_ex, A_im = RKTable("111")
    # stages, c, A_ex, A_im = RKTable("222")
    stages, c, A_ex, A_im = RKTable("443")

    u = zeros(Complex, nSteps)
    u0 = 1
    u[1] = u0

    for i=2:nSteps
        u[i] = RKStep(k, u[i-1], stages, c, A_ex, A_im, f, g)
    end        

    uExact = @. u0*exp((α + im*β)*t)

    du = u .- uExact
    println(norm(du))
    println(real(du[end]))
    println(imag(du[end]))

    plot(t, real(uExact), "k")
    plot(t, real(u), "r--")
    plot(t, imag(uExact), "k")
    plot(t, imag(u), "r--")
    tight_layout()
    savefig("u.png")
    println("u.png")
    plt.close()
end

run()