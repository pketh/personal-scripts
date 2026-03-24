#!/usr/bin/env python3
"""
Swimming Pool Temperature Estimator
====================================
Calculates whether a swimming pool in Charlotte, NC will be warm based on
the last 5 days of weather data, the pool's volume, and thermodynamic
principles.

Run:  python3 pool_temp.py
Deps: pip3 install requests   (only external dependency)
"""

import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

# ─── Pool Geometry ────────────────────────────────────────────────────────────
# Trapezoidal cross-section pool:
#
# Volume of a prism with trapezoidal cross-section:
#   V = width × length × (deep + shallow) / 2

POOL_LENGTH_FT = 41.0
POOL_WIDTH_FT = 20.0
DEEP_END_FT = 4.5
SHALLOW_END_FT = 3.5

POOL_VOLUME_FT3 = POOL_WIDTH_FT * POOL_LENGTH_FT * (DEEP_END_FT + SHALLOW_END_FT) / 2.0
POOL_VOLUME_LITERS = POOL_VOLUME_FT3 * 28.3168  # 1 ft³ ≈ 28.3168 L
POOL_VOLUME_GAL = POOL_VOLUME_FT3 * 7.48052     # for display

# ─── Thermodynamic Constants ─────────────────────────────────────────────────
WATER_DENSITY_KG_PER_L = 1.0
SPECIFIC_HEAT_WATER = 4186.0   # J/(kg·°C)
WATER_MASS_KG = POOL_VOLUME_LITERS * WATER_DENSITY_KG_PER_L

# Surface area of the pool (top, exposed to sun/air) in m²
POOL_SURFACE_AREA_M2 = (POOL_LENGTH_FT * POOL_WIDTH_FT) * 0.0929  # ft² → m²

# ─── Charlotte, NC Coordinates ───────────────────────────────────────────────
LATITUDE = 35.2271
LONGITUDE = -80.8431

# ─── "Warm" Threshold ────────────────────────────────────────────────────────
# The American Red Cross recommends 78 °F (25.6 °C) for recreational swimming.
WARM_THRESHOLD_C = 25.6  # °C  (≈ 78 °F)

# ─── Refresh Interval ────────────────────────────────────────────────────────
REFRESH_SECONDS = 3600  # 1 hour


def fetch_weather() -> dict:
    """
    Fetch the last 5 days of hourly weather from Open-Meteo (no API key needed).
    Returns hourly temperature_2m, relative_humidity_2m, windspeed_10m,
    direct_radiation, and precipitation.
    """
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,"
        f"direct_radiation,precipitation"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=America%2FNew_York"
        f"&past_days=0"
    )

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return data


def estimate_pool_temperature(weather: dict) -> float:
    """
    Walk through the hourly weather data and iteratively estimate the pool
    water temperature using a simplified energy-balance model.

    Energy inputs  (+):
        • Solar radiation absorbed by the water surface
        • Convective heat gain when air is warmer than water

    Energy losses  (−):
        • Evaporative cooling (latent heat)
        • Convective heat loss when air is cooler than water
        • Long-wave radiative cooling to the sky

    We assume the pool starts at the mean air temperature of the first day.
    """
    hourly = weather["hourly"]
    temps_c = hourly["temperature_2m"]           # °C
    humidity = hourly["relative_humidity_2m"]     # %
    wind_ms = hourly["windspeed_10m"]            # m/s (API returns km/h, but
    radiation = hourly["direct_radiation"]       # W/m²
    precip_mm = hourly["precipitation"]          # mm

    # Convert wind from km/h → m/s (Open-Meteo returns km/h by default)
    wind_ms = [w / 3.6 if w is not None else 0.0 for w in wind_ms]

    # Initialize pool temp to average air temp over first 24 hours
    first_day_temps = [t for t in temps_c[:24] if t is not None]
    if not first_day_temps:
        first_day_temps = [t for t in temps_c if t is not None]
    pool_temp = sum(first_day_temps) / len(first_day_temps) if first_day_temps else 20.0

    dt = 3600.0  # 1-hour time step in seconds
    A = POOL_SURFACE_AREA_M2
    m = WATER_MASS_KG
    cp = SPECIFIC_HEAT_WATER

    for i in range(len(temps_c)):
        t_air = temps_c[i]
        rh = humidity[i]
        w = wind_ms[i]
        rad = radiation[i]
        rain = precip_mm[i]

        if any(v is None for v in [t_air, rh, w, rad, rain]):
            continue

        # --- 1. Solar heat gain ---
        # ~70% of direct radiation is absorbed by water (rest reflected)
        solar_absorptivity = 0.70
        q_solar = solar_absorptivity * rad * A  # Watts

        # --- 2. Convective exchange (Newton's law of cooling) ---
        # h_conv ≈ 5.7 + 3.8·v  (W/m²·°C) — empirical for outdoor surfaces
        h_conv = 5.7 + 3.8 * w
        q_conv = h_conv * A * (t_air - pool_temp)  # + if air warmer

        # --- 3. Evaporative cooling ---
        # Saturation vapor pressure (Tetens formula) in kPa
        def sat_vp(t):
            return 0.6108 * math.exp((17.27 * t) / (t + 237.3))

        e_s_water = sat_vp(pool_temp)   # at water surface temp
        e_a = sat_vp(t_air) * rh / 100  # ambient vapor pressure

        # Evaporation rate (Penman-style simplified), kg/(m²·s)
        # E = (0.0313·w + 0.0128) × (e_s - e_a)  [approx]
        evap_rate = max(0.0, (0.0313 * w + 0.0128) * (e_s_water - e_a))
        latent_heat = 2.45e6  # J/kg  (latent heat of vaporization)
        q_evap = -evap_rate * latent_heat * A  # always a loss

        # --- 4. Long-wave radiative loss ---
        # Stefan-Boltzmann: pool surface radiates as ~0.95 emissivity body
        sigma = 5.67e-8
        emissivity = 0.95
        T_pool_K = pool_temp + 273.15
        T_sky_K = t_air + 273.15 - 20  # sky ~20 K cooler than air
        q_rad = -emissivity * sigma * A * (T_pool_K**4 - T_sky_K**4)

        # --- 5. Rain mixing (cools or warms slightly) ---
        # rain mm over 1 hour → liters per m² → total liters on pool surface
        rain_liters = rain * A  # mm × m² = liters
        rain_mass = rain_liters  # kg (density ≈ 1)
        if rain_mass > 0 and (m + rain_mass) > 0:
            # Assume rainwater is at the wet-bulb temperature ≈ air temp
            pool_temp = (pool_temp * m + t_air * rain_mass) / (m + rain_mass)
            # Don't permanently add rain mass (pool overflows)

        # --- Net energy balance ---
        q_net = q_solar + q_conv + q_evap + q_rad  # Watts
        delta_t = (q_net * dt) / (m * cp)           # °C change this hour
        pool_temp += delta_t

    return pool_temp


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def run_once() -> bool:
    """Fetch weather, calculate, print results, return is_warm boolean."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═' * 60}")
    print(f"  Swimming Pool Temperature Estimator")
    print(f"  Charlotte, NC  •  {now}")
    print(f"{'═' * 60}")

    # Pool info
    print(f"\n  Pool dimensions : {POOL_LENGTH_FT:.0f} ft × {POOL_WIDTH_FT:.0f} ft")
    print(f"  Depth           : {SHALLOW_END_FT:.0f} ft (shallow) → {DEEP_END_FT:.0f} ft (deep)")
    print(f"  Volume          : {POOL_VOLUME_FT3:.1f} ft³  ({POOL_VOLUME_GAL:.0f} gal / {POOL_VOLUME_LITERS:.0f} L)")
    print(f"  Water mass      : {WATER_MASS_KG:.0f} kg")
    print(f"  Specific heat   : {SPECIFIC_HEAT_WATER} J/(kg·°C)")
    print(f"  Warm threshold  : {WARM_THRESHOLD_C} °C  ({c_to_f(WARM_THRESHOLD_C):.1f} °F)")

    print(f"\n  Fetching last 5 days of weather data …", end=" ", flush=True)
    try:
        weather = fetch_weather()
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        return False

    n_hours = len(weather["hourly"]["temperature_2m"])
    print(f"OK ({n_hours} hourly data points)")

    # Summarize weather
    temps = [t for t in weather["hourly"]["temperature_2m"] if t is not None]
    if temps:
        print(f"  Air temp range  : {min(temps):.1f} – {max(temps):.1f} °C "
              f"({c_to_f(min(temps)):.0f} – {c_to_f(max(temps)):.0f} °F)")
        print(f"  Air temp avg    : {sum(temps)/len(temps):.1f} °C "
              f"({c_to_f(sum(temps)/len(temps)):.0f} °F)")

    # Estimate pool temperature
    pool_temp_c = estimate_pool_temperature(weather)
    pool_temp_f = c_to_f(pool_temp_c)
    is_warm = pool_temp_c >= WARM_THRESHOLD_C

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Estimated pool temp : {pool_temp_c:5.1f} °C ({pool_temp_f:5.1f} °F) │")
    print(f"  │  Warm threshold      : {WARM_THRESHOLD_C:5.1f} °C ({c_to_f(WARM_THRESHOLD_C):5.1f} °F) │")
    print(f"  │  IS WARM             : {'✅ YES' if is_warm else '❌ NO':>13s}       │")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  is_warm = {is_warm}")
    return is_warm


def main():
    print("Starting pool temperature monitor (Ctrl+C to quit) …")
    print(f"Checks every {REFRESH_SECONDS // 60} minutes.")

    while True:
        try:
            result = run_once()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            result = False

        print(f"\n  Next check in {REFRESH_SECONDS // 60} minutes …")
        try:
            time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            sys.exit(0)


if __name__ == "__main__":
    main()
