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
import pytensor.tensor as tt
from celerite2.theano import terms, GaussianProcess
from astropy import units as units, constants as const
import matplotlib as mpl
import platform
import arviz as az

def build_model(x,y,yerr,u_s,t0s,periods,rps,a_ps, texp,b_ps=0.62, P_rot=200, mask=None, start=None):
    nb_planet = 1
    planets_str = "b"
    t0s = np.array([t0s])
    periods = np.array([periods])
    rps = np.array([rps])
    a_ps = np.array([a_ps])
    #rors = rps

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Shared parameters
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # Stellar parameters.  These are usually determined from spectroscopy
        # and/or isochrone fits.
        logg_star = pm.Normal("logg_star", mu=4.45, sd=0.05)
        r_star = pm.Normal("r_star", mu=1.004, sd=0.018)

        # Limb-darkening: adopt Kipping 2013.
        #u_star = xo.distributions.QuadLimbDark("u_star", testval=u_s)
        u = xo.distributions.QuadLimbDark("u_star", testval=u_s)
        star = xo.LimbDarkLightCurve(u_s)

        # Orbital parameters for the planet.  Use mean values from Holczer+16.
        a = pm.Uniform("a", lower=a_ps-1,upper=a_ps+1, shape=nb_planet)
        b = pm.Uniform("b", lower=b_ps-0.1,upper=b_ps+0.1, shape=nb_planet)
       # b = xo.distributions.ImpactParameter("b", ror=rors, shape=nb_planet, testval=b_ps)
        t0 = pm.Normal("t0", mu=t0s, sd=0.005, shape=nb_planet)
        logP = pm.Normal("logP", mu=np.log(periods), sd=0.05, shape=nb_planet)
        period = pm.Deterministic("period", pm.math.exp(logP))
        log_depth = pm.Normal("log_depth", mu=np.log(rps**2), sd=1.5,shape=nb_planet)
        depth = pm.Deterministic("depth", tt.exp(log_depth))
        ror = pm.Deterministic(
            "ror",
            star.get_ror_from_approx_transit_depth(depth, b),
        )
        r_pl = pm.Deterministic("r_pl", ror * r_star)


        # Define the orbit model.
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            a=a,
            b=b,
        )

        transit_model = mean + tt.sum(
            star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp),
            axis=-1,
        )

        # Convenience function for plotting.
        pm.Deterministic(
            "transit_pred",
            star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp),
        )

        # Use the GP model from the stellar variability tutorial at
        # https://gallery.exoplanet.codes/en/latest/tutorials/stellar-variability/

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sd=2)

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1, 5)
        )
        # Rotation period is 200 days, from Lomb Scargle
        log_prot = pm.Normal("log_prot", mu=np.log(P_rot), sd=0.02)
        prot = pm.Deterministic("prot", tt.exp(log_prot))
        log_Q0 = pm.Normal("log_Q0", mu=0, sd=2)
        log_dQ = pm.Normal("log_dQ", mu=0, sd=2)
        f = pm.Uniform("f", lower=0.01, upper=1)

        # Set up the Gaussian Process model. See
        # https://celerite2.readthedocs.io/en/latest/tutorials/first/ for an
        # introduction. Here, we have a quasiperiodic term:
        kernel = terms.RotationTerm(
            sigma=sigma_rot,
            period=prot,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        #
        # Note mean of the GP is defined here to be zero, so our "observations"
        # will need to subtract the transit model.  The inverse choice could
        # also be made.
        #
        gp = GaussianProcess(
            kernel,
            t=x[mask],
            diag=yerr[mask] ** 2 + tt.exp(2 * np.log(np.mean(yerr))),
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("transit_obs", observed=y[mask] - transit_model)

        # Compute the GP model prediction for plotting purposes
        pm.Deterministic("gp_pred", gp.predict(y[mask] - transit_model))

        # Track planet radius in Jovian radii
       # r_planet = pm.Deterministic(
       #     "r_planet",
       #     (ror * r_star) * (1 * units.Rsun / (1 * units.Rjup)).cgs.value,
       # )
#
        # Optimize the MAP solution.
        if start is None:
            start = model.test_point

        map_soln = start

        map_soln = pmx.optimize(
            start=map_soln, vars=[sigma_rot, f, prot, log_Q0, log_dQ]
        )
        map_soln = pmx.optimize(
            start=map_soln,
            vars=[
                t0,
                a,
                b,
                period,
                mean,
                r_pl,
            ],
        )
        map_soln = pmx.optimize(start=map_soln)

    return model, map_soln



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
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(
        x[mask], gp_mod+1, color="C2", label="model without transit", zorder=5, lw=0.5
    )
    ax.legend(fontsize=10)
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
    ax.legend(fontsize=10, loc=3)
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
    ax.legend(fontsize=10)
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
