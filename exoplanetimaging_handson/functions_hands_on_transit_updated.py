import pandas as pd
import matplotlib.pyplot as plt
import batman
import time
import matplotlib.pyplot as plt
import corner
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc as pm
import pymc_ext as pmx
from pytensor import tensor as tt
from celerite2.pymc import terms, GaussianProcess
from astropy import units as units, constants as const
import matplotlib as mpl
import platform
import arviz as az

def build_model(x, y, yerr, u_s, t0s, periods, rps, a_ps, texp, b_ps=0.62, P_rot=200, mask=None, start=None):
    import pymc as pm
    import pymc_ext as pmx
    from pytensor import tensor as tt
    import exoplanet as xo
    from celerite2.pymc import terms, GaussianProcess
    import numpy as np

    nb_planet = 1
    planets_str = "b"
    t0s = np.array([t0s])
    periods = np.array([periods])
    rps = np.array([rps])
    a_ps = np.array([a_ps])

    phase_lc = np.linspace(-0.3, 0.3, 100)

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:
        # Shared parameters
        mean = pm.Normal("mean", mu=0.0, sigma=1.0)

        # Stellar parameters
        logg_star = pm.Normal("logg_star", mu=4.45, sigma=0.05)
        r_star = pm.Normal("r_star", mu=1.004, sigma=0.018)

        # Limb-darkening
        u_star = xo.distributions.QuadLimbDark("u_star", testval=u_s)
        star = xo.LimbDarkLightCurve(u_s)

        # Planet parameters
        a = pm.Uniform("a", lower=a_ps-1, upper=a_ps+1, shape=nb_planet)
        b = pm.Uniform("b", lower=b_ps-0.1, upper=b_ps+0.1, shape=nb_planet)
        t0 = pm.Normal("t0", mu=t0s, sigma=0.005, shape=nb_planet)
        logP = pm.Normal("logP", mu=np.log(periods), sigma=0.05, shape=nb_planet)
        period = pm.Deterministic("period", pm.math.exp(logP))
        log_depth = pm.Normal("log_depth", mu=np.log(rps**2), sigma=1.5, shape=nb_planet)
        depth = pm.Deterministic("depth", tt.exp(log_depth))
        ror = pm.Deterministic("ror", star.get_ror_from_approx_transit_depth(depth, b))
        r_pl = pm.Deterministic("r_pl", ror * r_star)
        # ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
        # ecc = pm.Deterministic("ecc", tt.sum(ecs**2))
        omega = pm.Uniform("omega", lower=0., upper=360., shape=nb_planet)
        ecc = xo.distributions.eccentricity.kipping13("ecc")
        # print(xo.eccentricity.kipping13("ecc", "omega"))

        # Orbit and transit
        orbit = xo.orbits.KeplerianOrbit(
            r_star=r_star,
            period=period,
            t0=t0,
            a=a,
            b=b,
            ecc=ecc,
            omega=omega,
        )
        transit_model = mean + tt.sum(star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp), axis=-1)
        pm.Deterministic("transit_pred", star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp))

        # Jitter
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=2)

        # # Rotation GP parameters
        # mean_val, std_val = 1, 5
        # alpha = (mean_val / std_val)**2 + 2
        # beta = mean_val * (alpha - 1)
        # sigma_rot = pm.InverseGamma("sigma_rot", alpha=alpha, beta=beta)
        # log_prot = pm.Normal("log_prot", mu=np.log(P_rot), sigma=0.02)
        # prot = pm.Deterministic("prot", tt.exp(log_prot))
        # log_Q0 = pm.Normal("log_Q0", mu=0, sigma=2)
        # log_dQ = pm.Normal("log_dQ", mu=0, sigma=2)
        # f = pm.Uniform("f", lower=0.01, upper=1)

        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sigma=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0, sigma=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sigma=10
        )

        # Compute the model residuals
        resid = y[mask] - transit_model

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc), mean=transit_model)
        gp.marginal("gp", observed=y[mask])
        
        # # GP with transit as mean
        # kernel = terms.RotationTerm(sigma=sigma_rot, period=prot, Q0=tt.exp(log_Q0), dQ=tt.exp(log_dQ), f=f)
        # gp = GaussianProcess(kernel, t=x[mask], diag=yerr[mask] ** 2, mean=transit_model, quiet=True)
        # gp.marginal("transit_obs", observed=y[mask])

        # Compute the GP model prediction for plotting purposes
        pm.Deterministic("gp_pred", gp.predict(resid))

        # Compute and save the phased light curve models
        pm.Deterministic(
            "lc_pred",
            1e3
            * star.get_light_curve(
                orbit=orbit, r=r_pl, t=t0 + phase_lc, texp=texp
            )[..., 0],
        )

        # Optimize MAP
        if start is None:
            start = model.initial_point()

        map_soln = start
        # map_soln = pmx.optimize(start=map_soln, vars=[t0, a, b, period, mean, r_pl])
        # map_soln = pmx.optimize(start=map_soln, vars=[sigma_rot, f, prot, log_Q0, log_dQ])
        # map_soln = pmx.optimize(start=map_soln, vars=[t0, a, b, period, mean, r_pl])
        # map_soln = pmx.optimize(start=map_soln, vars=[sigma_rot, f, prot, log_Q0, log_dQ])
        # map_soln = pmx.optimize(start=map_soln, vars=[t0, a, b, period, mean, r_pl])
        
        map_soln = pmx.optimize(
            start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
        map_soln = pmx.optimize(start=map_soln, vars=[logP, t0])
        map_soln = pmx.optimize(start=map_soln, vars=[u_star])
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
        map_soln = pmx.optimize(start=map_soln, vars=[ecs])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(
            start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(
            zip(
                ["light_curves", "gp_pred"],
                pmx.eval_in_model([light_curves, gp.predict(resid)], map_soln),
            )
        )

        
        # map_soln = pmx.optimize(start=map_soln, vars=[t0, a, b, period, mean, r_pl,sigma_rot, f, prot, log_Q0, log_dQ])

    return model, map_soln, extras



def plot_light_curve(x, y, yerr, soln, mask=None):
    planets_str = 'b'
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close("all")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]

    if len(x[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams["agg.path.chunksize"] = 10000

    ax.errorbar(
        x[mask],
        y[mask]+1,
        yerr=yerr,
        fmt='o',
        color="k",
        label="data",
        zorder=4,
    )
    gp_mod = soln["gp_pred"]  + soln["mean"]
    ax.plot(
        x[mask], gp_mod+1, color="C2", label="model without transit", zorder=5, lw=0.5
    )
    # ax.legend(fontsize=10)
    ax.set_ylabel("$f$")

    ax = axes[1]
    ax.plot(x[mask], y[mask] - gp_mod+1, ".k", label="data")
    for i, l in enumerate(planets_str):
        mod = soln["transit_pred"][:, i]
        ax.plot(
            x[mask],
            mod+1,
            label="planet {0} [model under]".format(l),
            zorder=5,
        )
    # ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("$f_\mathrm{dtr}$")

    #ax = axes[2]
    #ax.plot(x[mask], y[mask] - gp_mod+1, "k", label="data - MAPgp")
    #for i, l in enumerate(planets_str):
    #    mod = soln["transit_pred"][:, i]
    #    ax.plot(x[mask], mod+1, label="planet {0} [model over]".format(l))
    #ax.legend(fontsize=10, loc=3)
    #ax.set_ylabel("$f_\mathrm{dtr}$ [zoom]")
    #ymin = np.min(mod) - 0.05 * abs(np.min(mod))
    #ymax = abs(ymin)
    ##ax.set_ylim([ymin, ymax])

    ax = axes[2]
    mod = gp_mod + np.sum(soln["transit_pred"], axis=-1)
    ax.plot(x[mask], y[mask] - mod+1, ".k")
    ax.axhline(1, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals")
    ax.set_xlim(x[mask].min(), x[mask].max())
    ax.set_xlabel("time [days]")

    fig.tight_layout()

def run_sampling(model,map_estimate,tune=200,draws=100,cores=2):

    # Change this to "1" if you wish to run it.
    RUN_THE_SAMPLING = 1

    if RUN_THE_SAMPLING:
        with model:
            trace_with_gp = pm.sample(
                tune=tune,
                draws=draws,
                start=map_estimate,
                # Parallel sampling runs poorly or crashes on macos
                cores=1 if platform.system() == "Darwin" else 2,
                chains=2,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=[261136679, 261136680],
                init="adapt_full",
            )

        az.summary(
            trace_with_gp,
            var_names=[
                "mean",
                "t0",
                "a",
                "b",
                "r_pl",
                "period",
                "log_jitter",
                "sigma_rot",
                "log_prot",
                "log_Q0",
                "log_dQ",
            ],
        )
    return trace_with_gp

def plot_best_fit(x, y, yerr, soln, mask=None):
    planets_str = 'b'

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)


    if len(x[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams["agg.path.chunksize"] = 10000

    ax.errorbar(
        x[mask],
        y[mask]+1,
        yerr=yerr,
        fmt='o',
        color="k",
        alpha=0.4,
        label="data",
        zorder=3,
    )
    mod = np.zeros(len(soln["transit_pred"][:, 0]))
    for i, l in enumerate(planets_str):
        mod += soln["transit_pred"][:, i]
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(
        x[mask], gp_mod+1+mod, color="red", label="best fit transit model", zorder=4, lw=1
    )
    ax.plot(
        x[mask], gp_mod+1,'--', color="C2", label="without transit", zorder=5, lw=1
    )
    # ax.legend(fontsize=10)
    ax.set_ylabel("Relative flux")
    ax.set_xlabel("Time to mid-transit (days)")




def output_table(trace_with_gp,var_names):

    table = az.summary(trace_with_gp, var_names=var_names,round_to=5)
    return table

def acknowledgment():

    with pm.Model() as model:
        u = xo.distributions.QuadLimbDark("u")
        orbit = xo.orbits.KeplerianOrbit(period=10.0)
        light_curve = xo.LimbDarkLightCurve(u[0], u[1])
        transit = light_curve.get_light_curve(r=0.1, orbit=orbit, t=[0.0, 0.1])

        txt, bib = xo.citations.get_citations_for_model()
    print(txt)


def synthetic_error(file):
    filename = file+'.csv' #nom du fichier d'entrée, c'est à dire de la courbe de transit
    df_lc = pd.read_csv(filename,sep=',')
    df_lc = df_lc.sort_values(by="Time (days)")
    df_lc['Flux error'] = np.random.uniform(low=0.0001, high=0.0006,size=len(df_lc))
    df_lc.to_csv(file+'_bis.csv',sep=',',index=None)
