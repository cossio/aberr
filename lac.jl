import ArgParse, Iterators

nolessthan(a) = x -> x >= a

arg_parse_settings = ArgParse.ArgParseSettings(fromfile_prefix_chars='@')
@ArgParse.add_arg_table arg_parse_settings begin
    "--glc"
        help = "blood glc concentration, in mM"
        arg_type = Float64
        required = true
        range_tester = nolessthan(0)
    "--flow"
        help = "blood flow velocity, in decimeters/day"
        arg_type = Float64
        required = true
        range_tester = nolessthan(0)
    "--glc-initial"
        help = "initial glucose concentration per site, in mM"
        arg_type = Float64
        default = 2.0
        range_tester = nolessthan(0)
    "--alpha"
        help = "glucose uptake factor for cancer cells"
        arg_type = Float64
        default = 1e2
        range_tester = nolessthan(1)
    "--size"
        help = "number of sites in grid = size x size"
        arg_type = Int
        default = 20
        range_tester = nolessthan(0)
    "--dx"
        help = "separation between cells, in dm"
        arg_type = Float64
        default = 1e-4
        range_tester = nolessthan(0)
    "--steps"
        help = "number of steps to simulate"
        arg_type = Int
        default = 5 * 10^6
        range_tester = nolessthan(0)
    "--T"
        help = "time to simulate, in days"
        arg_type = Float64
        default = 50.
        range_tester = nolessthan(0)
    "--write-interval"
        help = "steps between output file writes"
        arg_type = Int
        default = 10^4
        range_tester = nolessthan(0)
    "--dt"
        help = "time step, in days"
        arg_type = Float64
        default = 1e-6
        range_tester = nolessthan(0)
    "--diffusion-glc"
        help = "glc diffusion coefficient, in decimeters^2/day"
        arg_type = Float64
        default = 1.4e-3
        range_tester = nolessthan(0)
    "--diffusion-lac"
        help = "lac diffusion coefficient, in decimeters^2/day"
        arg_type = Float64
        default = 1.4e-3
        range_tester = nolessthan(0)
    "--max-ox"
        help = "max ox flow, in pmol/day/cel"
        arg_type = Float64
        default = 6.48
        range_tester = nolessthan(0)
    "--Kglc"
        help = "glc uptake Michaelis constant, in mM"
        arg_type = Float64
        default = 1.0
        range_tester = nolessthan(0)
    "--Klac"
        help = "lac uptake Michaelis constant, in mM"
        arg_type = Float64
        default = 4.68
        range_tester = nolessthan(0)
    "--Vglc"
        help = "glc max uptake rate, in pmol/day/cel"
        arg_type = Float64
        default = 0.5
        range_tester = nolessthan(0)
    "--Vlac"
        help = "lac max uptake rate, in pmol/day/cel"
        arg_type = Float64
        default = 1e2
        range_tester = nolessthan(0)
    "--mu-max"
        help = "max μ, in day^-1"
        arg_type = Float64
        default = 3.0
        range_tester = nolessthan(0)
    "--delta-max"
        help = "max δ, in day^-1"
        arg_type = Float64
        default = 2.0
        range_tester = nolessthan(0)
    "--Kmu"
        help = "Michaelis constant for lac inhibition of μ, in mM"
        arg_type = Float64
        default = 8.0
        range_tester = nolessthan(0)
    "--Kdelta"
        help = "Michaelis constant for lac enhancement of δ, in mM"
        arg_type = Float64
        default = 15.0
        range_tester = nolessthan(0)
    "--hill"
        help = "Hill constant for lactate effects on μ and δ"
        arg_type = Float64
        default = 1.0
        range_tester = nolessthan(0)
    "--A"
        help = "A = fatp / μ, in pmol/cel"
        arg_type = Float64
        default = 10.0
        range_tester = nolessthan(0)
    "--ATPm"
        help = "ATP maintenance flow, in pmol/day/cel"
        arg_type = Float64
        default = 7.7
        range_tester = nolessthan(0)
    "--out"
        help = "path suffix to output simulation matrices"
        arg_type = String
        required = true
        range_tester = x -> dirname(x) == "" || ispath(dirname(x))
end
args = ArgParse.parse_args(arg_parse_settings)

michaelis_menten_inhibition(c::Float64, K::Float64) = K / (c + K)
michaelis_menten_inhibition(c::Float64, K::Float64, h::Float64) = michaelis_menten_inhibition(c^h, K^h)
michaelis_menten(c::Float64, K::Float64, V::Float64) = V * c / (K + c)
michaelis_menten(c::Float64, K::Float64, V::Float64, h::Float64) = michaelis_menten(c^h, K^h, V)

grid_at(matrix::Matrix, i::Integer, j::Integer) = matrix[mod1(i, size(matrix, 1)), mod1(j, size(matrix, 2))]

function grid_at!{T}(matrix::Matrix{T}, i::Integer, j::Integer, v::T)
    matrix[mod1(i, size(matrix, 1)), mod1(j, size(matrix, 2))] = v
end


function laplacian(matrix::Matrix{Float64}, i::Int, j::Int, dx::Float64)
    neighbors_sum::Float64 = grid_at(matrix, i - 1, j) + grid_at(matrix, i, j - 1) +
                             grid_at(matrix, i + 1, j) + grid_at(matrix, i, j + 1)
    return (4 / dx^2) * (neighbors_sum / 4 - matrix[i,j])
end

function update_concentrations!(concentrations::Matrix{Float64}, diffusion::Matrix{Float64}, diffusion_coefficient::Float64, uptakes::Matrix{Float64},
                                blood_concentration::Float64, flow::Float64, dt::Float64, dx::Float64)

    @assert size(concentrations) == size(diffusion) == size(uptakes)
    M, N = size(concentrations)

    for i = 1:M, j = 1:N
        diffusion[i,j] = diffusion_coefficient * laplacian(concentrations, i, j, dx)
    end
    @assert abs(sum(diffusion)) ≤ 1e-6 * sum(concentrations)

    const ν::Float64 = dx^3

    Δ::Float64 = 0.0

    for i = 1:M, j = 1:N
        old_value::Float64 = concentrations[i,j]
        concentrations[i,j] += (diffusion[i,j] - uptakes[i,j] / ν + flow / dx * (blood_concentration - concentrations[i,j])) * dt
        concentrations[i,j] = max(0.0, concentrations[i,j])
        if concentrations[i,j] ≠ old_value
            Δ = max(Δ, abs(concentrations[i,j] - old_value) / max(concentrations[i,j], old_value))
        end
    end

    return Δ
end

import Base.mean
"""
    mean(f::Function, A::AbstractArray{T,N})

Compute the mean of the elements f(x) with x ∈ A.
"""
mean(f::Function, A::AbstractArray) = sum(f, A) / length(A)

function grow(mut::Matrix{Bool}, i::Int, j::Int)
    @assert mut[i,j]
    Δi::Int = rand([0 rand([-1,1])])
    Δj::Int = Δi == 0 ? rand([-1,1]) : 0
    @assert Δi * Δj == 0 && abs(Δi + Δj) == 1
    for s = 1:4
        grid_at(mut, i + Δi, j + Δj) || break
        Δi, Δj = -Δj, Δi
    end
    @assert grid_at(mut, i + Δi, j + Δj) == false
    grid_at!(mut, i + Δi, j + Δj, true)
end

function main()

    const L::Int = args["size"]
    const Katp::Float64 = args["A"] * args["mu-max"]
    const Kglc::Float64 = args["Kglc"] * 1e9
    const Klac::Float64 = args["Klac"] * 1e9
    const Kμ::Float64 = args["Kmu"] * 1e9
    const Kδ::Float64 = args["Kdelta"] * 1e9
    const blood_glc::Float64 = args["glc"] * 1e9
    const glc₀::Float64 = args["glc-initial"] * 1e9

    glc::Matrix{Float64} = fill(glc₀, L, L)
    lac::Matrix{Float64} = zeros(L, L)
    diffusion::Matrix{Float64} = zeros(L, L)
    uglc::Matrix{Float64} = zeros(L, L)
    ulac::Matrix{Float64} = zeros(L, L)
    atp::Matrix{Float64} = zeros(L, L)
    μ::Matrix{Float64} = zeros(L, L)
    δ::Matrix{Float64} = zeros(L, L)
    ox::Matrix{Float64} = zeros(L, L)
    mut::Matrix{Bool} = fill(false, L, L)
    mut[div(L,2), div(L,2)] = true

    glc_file = open(string(args["out"], ".glc"), "w")
    lac_file = open(string(args["out"], ".lac"), "w")
    mut_file = open(string(args["out"], ".mut"), "w")
    log_file = open(string(args["out"], ".log"), "w")

    t::Float64 = 0.0
    s::Int = 0

    while true
        # update fluxes
        for i = 1:L, j = 1:L
            v_glc = michaelis_menten(glc[i,j], Kglc, args["Vglc"])
            v_lac = michaelis_menten(lac[i,j], Klac, args["Vlac"])
            if mut[i,j]
                uglc[i,j] = args["alpha"] * v_glc
                ox[i,j] = min(2uglc[i,j], args["max-ox"])
            else
                uglc[i,j] = min(v_glc, args["ATPm"] / 38)
                ox[i,j] = 2uglc[i,j] + min((args["ATPm"] - 38uglc[i,j]) / 18, v_lac)
            end
            atp[i,j] = 2uglc[i,j] + 18ox[i,j]
            ulac[i,j] = ox[i,j] - 2uglc[i,j]
            μ[i,j] = michaelis_menten(max(0.0, atp[i,j] - args["ATPm"]), Katp, args["mu-max"]) * michaelis_menten_inhibition(lac[i,j], Kμ, args["hill"])
            δ[i,j] = michaelis_menten(lac[i,j], Kδ, args["delta-max"], args["hill"])
        end

        # cancer boundary
        is_boundary(i::Int, j::Int) = mut[i,j] ≠ grid_at(mut, i - 1, j) || mut[i,j] ≠ grid_at(mut, i + 1, j) || mut[i,j] ≠ grid_at(mut, i, j + 1) || mut[i,j] ≠ grid_at(mut, i, j - 1)
        cancer_boundary::Vector{Tuple{Int,Int}} = collect(filter(x -> mut[x...] && is_boundary(x...), Iterators.product(1:L, 1:L)))

        if mod(s, args["write-interval"]) == 0 || s ≥ args["steps"] || t ≥ args["T"]
            # print and save stuff
            write(glc_file, "step $s\n")
            write(lac_file, "step $s\n")
            write(mut_file, "step $s\n")

            writedlm(glc_file, glc)
            writedlm(lac_file, lac)
            writedlm(mut_file, 1 * mut)

            status_line = "step $s \t time $t \t mut $(sum(mut)) \t glc $(mean(glc)) \t lac $(mean(lac)) \t LDH $(mean(x -> max(0, -x), ulac)) \t LDHr $(mean(x -> max(0, x), ulac))\n"
            write(log_file, status_line)
            print(status_line)

            if isempty(cancer_boundary) || length(cancer_boundary) == L * L || s ≥ args["steps"] || t ≥ args["T"]
                exit()
            end
        end

        # update concentrations
        Δglc::Float64 = update_concentrations!(glc, diffusion, args["diffusion-glc"], uglc, blood_glc, args["flow"], args["dt"], args["dx"])
        Δlac::Float64 = update_concentrations!(lac, diffusion, args["diffusion-glc"], ulac, 0.0      , args["flow"], args["dt"], args["dx"])
        Δmet::Float64 = max(Δglc, Δlac)

        # growth dynamics
        shuffle!(cancer_boundary)
        s += 1

        if !isempty(cancer_boundary) && length(cancer_boundary) < L * L && Δmet < 1e-6
            # met concentrations converged, do Gillespie step
            a₀::Float64 = sum(x -> μ[x...] + δ[x...], cancer_boundary)
            a::Float64 = 0.0
            r1::Float64, r2::Float64 = rand(2)
            t += 1 / a₀ * log(1 / r1)
            for (i,j) in cancer_boundary
                a += μ[i,j] + δ[i,j]
                if a > r2 * a₀
                    if a - δ[i,j] > r2 * a₀
                        grow(mut, i, j)
                    else
                        mut[i,j] = false
                    end
                    break
                end
            end
        else
            # met concentrations have not converged, do normal step
            t += args["dt"]
            for (i,j) in cancer_boundary
                @assert μ[i,j] + δ[i,j] ≤ 1 / args["dt"]
                r::Float64 = rand()
                r < μ[i,j] * args["dt"] && is_boundary(i,j) && grow(mut, i, j)
                μ[i,j] * args["dt"] ≤ r < (μ[i,j] + δ[i,j]) * args["dt"] && is_boundary(i,j) && (mut[i,j] = false)
            end
        end
    end

    close(log_file)
    close(glc_file)
    close(lac_file)
    close(mut_file)

end

main()
