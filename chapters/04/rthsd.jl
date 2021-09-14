using
    CSV,
    Random,
    ArgParse,
    Statistics,
    DataFrames,
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
workers = nthreads()

#
# data aquisition
#
df = DataFrame(CSV.File(read(stdin)))
df = unstack(df, :topic, :system, :score)
select!(df, Not(:topic))

nsystems = ncol(df)
raw = Matrix{Float64}(df)

#
# system means
#
sysmeans = Matrix{Float64}(undef, nsystems, nsystems)
for (i, j) in combinations(collect(enumerate(mean(raw; dims=1))), 2)
    # diagonal is the mean
    for (m, n) in (i, j)
        sysmeans[m,m] = n
    end

    # off-diagonals are differences
    ((a, x), (b, y)) = (i, j)
    sysmeans[a,b] = sysmeans[b,a] = abs(x - y)
end

#
# initialize counts
#
counts = zeros(Int, nsystems, nsystems, workers)

#
# create views of the data
#
data  = Array{Float64}(undef, nsystems, nrow(df), workers)
data .= transpose(raw)

#
# begin!
#
@info "Exploring:" args["B"] / factorial(ncol(df)) ^ nrow(df)

@threads for i in 1:args["B"]
    this = threadid()
    ptr = view(data,:,:,this)

    data[:,:,this] = mapslices(shuffle!, ptr; dims=1)

    m = mean(ptr; dims=2)
    d = maximum(m) - minimum(m)
    for (j, k) in combinations(1:nsystems, 2)
        if d >= sysmeans[j,k]
            counts[j,k,this] += 1
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
    pvalue = sum(sum, (counts[m,n,:], counts[n,m,:])) / args["B"]
    (lhs, rhs) = mean.(eachcol(df[!, [m,n]]))
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
