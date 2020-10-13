using
    CSV,
    Random,
    ArgParse,
    Statistics,
    DataFrames,
    SharedArrays,
    Base.Threads,
    Combinatorics

function cliargs()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--B"
        help = "Number of replications"
        arg_type = Int
    end

    return parse_args(s)
end

args = cliargs()

#
# data aquisition
#
df = CSV.File(read(stdin)) |> DataFrame!
df = unstack(df, :topic, :system, :score)
select!(df, Not(:topic))

nsystems = ncol(df)
raw = Matrix{Float64}(df)

#
# system means
#
sysmeans = Matrix{Float64}(undef, nsystems, nsystems)
for (i, j) in combinations(collect(enumerate(mean(raw; dims=1))), 2)
    for (m, n) in (i, j)
        sysmeans[m, m] = n
    end
    ((a, x), (b, y)) = (i, j)
    sysmeans[a, b] = sysmeans[b, a] = abs(x - y)
end
avgs = Matrix{Float64}(undef, nthreads(), nsystems)

#
# initialize counts
#
counts = SharedMatrix{Int}((nsystems, nsystems); init=0)

#
# create views of the data
#
data  = Array{Float64}(undef, nsystems, nrow(df), nthreads())
data .= transpose(raw)

#
# begin!
#
@info "Exploring:" args["B"] / 2 ^ nrow(df)

@threads for i in 1:args["B"]
    this = threadid()
    ptr = @view data[:,:,this]

    for (j, c) in enumerate(eachcol(ptr))
        shuffle!(c)
        data[:,j,this] = c
    end

    m = mean(ptr; dims=2)
    d = reduce(-, map(x -> x(m), (maximum, minimum)))
    for (j, k) in shuffle.(combinations(1:nsystems, 2))
        if d >= sysmeans[j, k]
            counts[j, k] += 1
        end
    end
end

#
# report
#
cnames = [
    :S1_name,
    :S2_name,
    :S1_mean,
    :S2_mean,
    :difference,
    :pvalue,
]
results = Matrix{Any}(undef, binomial(nsystems, 2), length(cnames))
sysnames = names(df)
for (i, (m, n)) in enumerate(combinations(1:nsystems, 2))
    pvalue = (counts[m, n] + counts[n, m]) / args["B"]
    (lhs, rhs) = mean.(eachcol(df[!, [m, n]]))
    results[i,:] = [
        sysnames[m],
        sysnames[n],
        lhs,
        rhs,
        lhs - rhs,
        pvalue,
    ]
end

results = DataFrame(results, cnames)
CSV.write(stdout, results)