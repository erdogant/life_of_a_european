import warnings
from bisect import bisect_right

import numpy as np
import pandas as pd

SLEEP_BOUNDS = (4 * 60, 14 * 60)   # minutes
ACTIVITY_BOUNDS = (1, 24 * 60)     # minutes

# States that simply continue until bedtime when their sampled duration ends
# early, instead of switching to "Other".
DEFAULT_EXTEND_STATES = ("Employment", "Study", "Volunteer work")

# Data prep step: evening activities are relabelled by where people are.
# Household chores, media and hobbies happen at home; socialising is "Leisure".
# (Both the short names from ACTIVITY_MAPPING and the long Eurostat names.)
DEFAULT_STATE_ALIASES = {
    "Household": "Home",
    "Media": "Home",
    "Hobbies": "Home",
    "Socialising": "Leisure",
    "Household and family care": "Home",
    "Mass media and reading": "Home",
    "Hobbies and computing": "Home",
    "Socialising and leisure": "Leisure",
}


def _with_size(person, base_size, eating_size, decay_min, step_min):
    """
    Turn one person's chronological [(timestamp, state), ...] into rows with a size.

    Size is `eating_size` while eating, then shrinks linearly back to `base_size`
    over `decay_min` minutes after eating ends. The shrinking is sampled every
    `step_min` minutes with extra rows that repeat the person's current state.
    The last item must be the final event (bedtime); no rows are added after it.
    """
    times = [t for t, _ in person]
    end = times[-1]

    eating = [(times[i], times[i + 1]) for i, (_, s) in enumerate(person[:-1]) if s == "Eating"]

    sample_times = set(times)
    if decay_min > 0:
        n_steps = int(np.ceil(decay_min / step_min))
        for _, eat_end in eating:
            for j in range(1, n_steps + 1):
                ts = eat_end + pd.Timedelta(minutes=min(j * step_min, decay_min))
                if ts < end:
                    sample_times.add(ts)

    def size_at(ts):
        for a, b in eating:
            if a <= ts < b:
                return eating_size
        past_ends = [b for _, b in eating if b <= ts]
        if past_ends and decay_min > 0:
            frac = (ts - max(past_ends)) / pd.Timedelta(minutes=decay_min)
            if frac < 1:
                return eating_size - (eating_size - base_size) * frac
        return base_size

    rows = []
    for ts in sorted(sample_times):
        state = person[bisect_right(times, ts) - 1][1]
        rows.append((ts, state, round(float(size_at(ts)), 3)))
    return rows


def _simulate_country(
    models,
    country,
    k,
    uid_start,
    start_date,
    seed,
    bedtime_mean_min,
    bedtime_sd_min,
    late_fraction,
    late_shift_min,
    extend_states,
    base_size,
    eating_size,
    size_decay_min,
    size_step_min,
    evening_travel_mean_min,
    evening_travel_sd_min,
    home_state,
    evening_activities,
    state_aliases,
    morning_travel_mean_min,
    morning_travel_sd_min,
    morning_travel_max_min,
    morning_min_prep_min,
):
    """Simulate k people for one country. uids run from uid_start to uid_start + k - 1."""
    if k < 1:
        raise ValueError(f"k must be at least 1 (country {country!r}).")
    if country not in models:
        raise KeyError(f"Country {country!r} not found in models.")

    percentage_models = models[country]["percentage"]
    time_models = models[country]["time"]

    if "Sleeping" not in time_models:
        raise KeyError(f"No sleeping-duration model found for {country!r}.")

    if seed is not None:
        # The fitted distributions draw from NumPy's global RNG.
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    preferred_order = ["Personal care", "Eating", "Travel", "Employment", "Study", "Volunteer work"]

    # Morning/daytime chain: the fixed order above.
    activities = [a for a in preferred_order
                  if a in percentage_models and a in time_models]

    # Activities that come before the morning trip (Personal care, Eating).
    pre_travel = set(activities[:activities.index("Travel")]) if "Travel" in activities else set()

    # Evening activities (household, leisure, ...): every other modelled activity
    # unless a list is given. Adding a code to ACTIVITY_MAPPING is enough.
    if evening_activities is None:
        evening = [a for a in percentage_models
                   if a not in preferred_order and a not in ("Sleeping", "Other")
                   and a in time_models]
    else:
        evening = [a for a in evening_activities
                   if a in percentage_models and a in time_models]

    def sample_minutes(distribution, lo, hi, tries=20):
        for _ in range(tries):
            value = float(np.asarray(distribution.generate(1)).reshape(-1)[0])
            if np.isfinite(value) and lo <= value <= hi:
                return int(round(value))

        # Explicit fallback: clip the last finite draw (or the midpoint).
        value = value if np.isfinite(value) else (lo + hi) / 2
        return int(round(min(max(value, lo), hi)))

    # 1. Who participates in each activity (indices 0..k-1, shifted to uids later).
    participants = {}
    for activity in activities + evening:
        pct = np.asarray(percentage_models[activity].generate(k), dtype=float).reshape(-1)
        pct = pct[np.isfinite(pct)]
        if len(pct) == 0:
            raise ValueError(f"Percentage model for {activity!r} returned no valid values.")
        n_part = int(round(k * np.clip(pct.mean(), 0, 100) / 100))
        participants[activity] = set(rng.choice(k, size=n_part, replace=False).tolist())

    # 2. Per-person timelines.
    base_date = pd.Timestamp(start_date).normalize()
    rows = []

    for i in range(k):
        uid = uid_start + i
        person = []   # [(timestamp, state)]

        shift = late_shift_min if rng.random() < late_fraction else 0
        # Not wrapped: a 00:30 bedtime stays on the evening of start_date
        # (i.e. next calendar day) instead of jumping back before everyone else.
        bedtime_raw = int(round(rng.normal(bedtime_mean_min + shift, bedtime_sd_min)))
        bedtime_min = bedtime_raw % (24 * 60)
        bedtime = base_date + pd.Timedelta(minutes=bedtime_raw)
        person.append((bedtime, "Sleeping"))

        sleep = sample_minutes(time_models["Sleeping"], *SLEEP_BOUNDS)
        wakeup = bedtime + pd.Timedelta(minutes=sleep)
        wake_day = wakeup.normalize()

        # Morning trip departure: drawn from N(mean, sd) but never later than
        # morning_travel_max_min (redrawn, so there is no spike at the limit).
        morning_start = None
        if morning_travel_mean_min is not None and i in participants.get("Travel", ()):
            lo = 5 * 60
            hi = morning_travel_max_min if morning_travel_max_min is not None else 24 * 60
            for _ in range(50):
                draw = int(round(rng.normal(morning_travel_mean_min, morning_travel_sd_min)))
                if lo <= draw <= hi:
                    break
            else:
                draw = min(max(draw, lo), hi)
            morning_start = wake_day + pd.Timedelta(minutes=draw)

            # Alarm clock: someone who would still be asleep (or not ready) at
            # departure wakes up early enough to have morning_min_prep_min to get
            # ready. Sleep is cut, but never below SLEEP_BOUNDS[0].
            latest_wake = morning_start - pd.Timedelta(minutes=morning_min_prep_min)
            earliest_wake = bedtime + pd.Timedelta(minutes=SLEEP_BOUNDS[0])
            if wakeup > latest_wake:
                wakeup = max(earliest_wake, latest_wake)

        # Next bedtime is always within 24 h of wake-up, so it ends the day.
        next_bedtime = wake_day + pd.Timedelta(minutes=bedtime_min)
        if next_bedtime <= wakeup:
            next_bedtime += pd.Timedelta(days=1)

        # Evening trip: people who travel in the morning travel again around
        # evening_travel_mean_min (same duration distribution) and then go home.
        evening_start = None
        if "Travel" in time_models and i in participants.get("Travel", ()):
            candidate = wake_day + pd.Timedelta(
                minutes=int(round(rng.normal(evening_travel_mean_min, evening_travel_sd_min)))
            )
            # Only if it fits in the day (>= 2 h after waking, >= 1 h before bed).
            if wakeup + pd.Timedelta(hours=2) <= candidate <= next_bedtime - pd.Timedelta(hours=1):
                evening_start = candidate

        # The daytime schedule ends when the evening trip starts (or at bedtime).
        day_end = evening_start if evening_start is not None else next_bedtime

        # Departure is a deadline for what comes before it (cut off, not shifted).
        # Anyone who wakes after it just follows the normal chain.
        deadline = morning_start if (morning_start is not None and morning_start > wakeup) else None

        cursor = wakeup
        last_state = None
        for activity in activities:
            if i not in participants[activity]:
                continue
            if cursor >= day_end:
                break
            if activity == last_state:
                continue   # never repeat the state a person is already in

            if activity == "Travel" and morning_start is not None and morning_start > cursor:
                if morning_start >= day_end:
                    continue   # no room for a morning trip before the evening one
                # Ready early: at home until it is time to leave.
                person.append((cursor, home_state))
                cursor = morning_start

            limit = day_end
            if deadline is not None and activity in pre_travel:
                if cursor >= deadline:
                    continue   # out of time before leaving
                limit = min(day_end, deadline)

            duration = sample_minutes(time_models[activity], *ACTIVITY_BOUNDS)
            person.append((cursor, activity))
            last_state = activity

            available = (limit - cursor).total_seconds() / 60
            if duration >= available:
                cursor = limit
                if limit == day_end:
                    break
                continue
            cursor += pd.Timedelta(minutes=duration)

        # Remaining time before the evening trip / bedtime: stay in the last
        # state if it may continue (e.g. Employment, Study); otherwise "Other".
        # Evening: trip home (if any), then evening activities in random order
        # until bedtime, then Home for whatever time is left.
        if evening_start is not None:
            # Gap before the trip: out somewhere, unless the last activity stretches.
            if cursor < day_end and last_state not in extend_states:
                person.append((cursor, "Other"))
            travel = sample_minutes(time_models["Travel"], *ACTIVITY_BOUNDS)
            if person[-1][1] != "Travel":
                person.append((evening_start, "Travel"))
            trip_end = evening_start + pd.Timedelta(minutes=travel)
            tail_start = trip_end if trip_end < next_bedtime else None
        elif cursor < next_bedtime and last_state not in extend_states:
            tail_start = cursor
        else:
            tail_start = None

        if tail_start is not None:
            t = tail_start
            mine = [a for a in evening if i in participants[a]]
            for j in rng.permutation(len(mine)):
                activity = mine[j]
                if t >= next_bedtime:
                    break
                duration = sample_minutes(time_models[activity], *ACTIVITY_BOUNDS)
                person.append((t, activity))
                available = (next_bedtime - t).total_seconds() / 60
                if duration >= available:
                    t = next_bedtime
                    break
                t += pd.Timedelta(minutes=duration)
            if t < next_bedtime:
                person.append((t, home_state))

        person.append((next_bedtime, "Sleeping"))

        # Relabel states (e.g. Household/Media/Hobbies -> Home) and merge
        # consecutive identical states into one.
        merged = []
        for ts, state in person:
            state = state_aliases.get(state, state)
            if merged and merged[-1][1] == state:
                continue
            merged.append((ts, state))
        person = merged

        for ts, state, size in _with_size(
            person, base_size, eating_size, size_decay_min, size_step_min
        ):
            rows.append({
                "datetime": ts, "uid": uid, "state": state,
                "country": country, "size": size,
            })

    return rows


def simulate_activity_log(
    models,
    countries="Netherlands",
    k=100,
    start_date="2000-01-01",
    seed=None,
    bedtime_mean_min=22 * 60,
    bedtime_sd_min=45,
    late_fraction=0.15,
    late_shift_min=90,
    extend_states=DEFAULT_EXTEND_STATES,
    base_size=1.0,
    eating_size=2.0,
    size_decay_min=60,
    size_step_min=5,
    evening_travel_mean_min=17 * 60,
    evening_travel_sd_min=30,
    home_state="Home",
    evening_activities=None,
    state_aliases=DEFAULT_STATE_ALIASES,
    morning_travel_mean_min=7 * 60 + 30,
    morning_travel_sd_min=30,
    morning_travel_max_min=8 * 60,
    morning_min_prep_min=15,
):
    """
    Simulate continuous activity timelines for people in one or more countries.

    Parameters
    ----------
    countries : str, list of str, or "all"
        Country name(s) as used in `models`. "all" uses every country that has
        a sleeping model (others are skipped with a warning).
    k : int or dict
        People per country. An int applies to every country; a dict such as
        {"Netherlands": 100, "Spain": 50} sets each country separately.
    seed : int or None
        Each country gets its own derived seed (seed + position in the list).
    bedtime_*, late_* :
        Bedtime is a mixture of two normals: most people draw from
        N(bedtime_mean_min, bedtime_sd_min); a fraction `late_fraction` is
        shifted `late_shift_min` later.
    extend_states : iterable of str
        If one of these is a person's last scheduled activity of the day, it
        continues until bedtime (its sampled duration is stretched) instead of
        switching to "Other". Pass () to always use "Other".
    base_size, eating_size :
        Size of a person normally, and while eating.
    size_decay_min, size_step_min :
        After eating ends, size shrinks linearly back to base_size over
        size_decay_min minutes, written as one row every size_step_min minutes
        (these rows repeat the person's current state).

    morning_travel_mean_min, morning_travel_sd_min, morning_travel_max_min :
        Departure of the morning trip is drawn from N(mean, sd) (default 07:30,
        SD 30 min) and redrawn until it is no later than morning_travel_max_min
        (default 08:00; None = no limit). Everyone who travels leaves at that time:
        personal care and eating are cut off at departure, anyone ready earlier
        is at home until then, and anyone who would still be asleep wakes up
        early enough to have morning_min_prep_min (default 15) to get ready (their
        sleep is shortened, never below 4 h). mean=None disables all of this.
    evening_travel_mean_min, evening_travel_sd_min :
        People who travel in the morning start a second trip around this time
        of day (default 17:00, SD 30 min). Its duration comes from the same
        Travel distribution. The daytime activities are cut off when it starts.
    evening_activities : list of str or None
        Activities done after the evening trip, in random order per person, until
        bedtime. None = every modelled activity that is not in the daytime chain
        (so new codes added to ACTIVITY_MAPPING are picked up automatically).
    state_aliases : dict or None
        Relabels states after scheduling, e.g. {"Household": "Home"}. By default
        Household, Media and Hobbies (done at home) become "Home" and Socialising
        becomes "Leisure"; Sports stays "Sports". Pass {} to keep the raw names.
    home_state :
        Final state of the day: once all sampled activities have played out
        (after the evening trip, or when there is no trip) people are at home
        until bedtime (default "Home").

    Duration policy: a sampled duration outside its plausible bounds is
    re-drawn (up to 20 times) and then clipped to the nearest bound.

    Returns
    -------
    DataFrame with columns datetime, uid, state, country, size. A row is
    written when a state starts or when the size changes. `uid` is unique
    across all countries.
    """
    skip_missing = countries == "all"
    if countries == "all":
        countries = [c for c in models if "Sleeping" in models[c]["time"]]
    elif isinstance(countries, str):
        countries = [countries]
    else:
        countries = list(countries)

    if not countries:
        raise ValueError("No countries to simulate.")
    if size_step_min <= 0:
        raise ValueError("size_step_min must be positive.")

    extend_states = set(extend_states or ())

    rows = []
    uid_start = 0

    for pos, country in enumerate(countries):
        k_country = k[country] if isinstance(k, dict) else k
        try:
            country_rows = _simulate_country(
                models=models,
                country=country,
                k=k_country,
                uid_start=uid_start,
                start_date=start_date,
                seed=None if seed is None else seed + pos,
                bedtime_mean_min=bedtime_mean_min,
                bedtime_sd_min=bedtime_sd_min,
                late_fraction=late_fraction,
                late_shift_min=late_shift_min,
                extend_states=extend_states,
                base_size=base_size,
                eating_size=eating_size,
                size_decay_min=size_decay_min,
                size_step_min=size_step_min,
                evening_travel_mean_min=evening_travel_mean_min,
                evening_travel_sd_min=evening_travel_sd_min,
                home_state=home_state,
                evening_activities=evening_activities,
                state_aliases=dict(state_aliases or {}),
                morning_travel_mean_min=morning_travel_mean_min,
                morning_travel_sd_min=morning_travel_sd_min,
                morning_travel_max_min=morning_travel_max_min,
                morning_min_prep_min=morning_min_prep_min,
            )
        except KeyError as err:
            if not skip_missing:
                available = sorted(models)
                have = sorted(models.get(country, {}).get("time", {}))
                raise KeyError(
                    f"{err} | countries in models: {available} | "
                    f"time models for {country!r}: {have}"
                ) from err
            warnings.warn(f"Skipping {country!r}: {err}")
            continue

        rows.extend(country_rows)
        uid_start += k_country

    result = pd.DataFrame(rows, columns=["datetime", "uid", "state", "country", "size"])
    result["datetime"] = pd.to_datetime(result["datetime"])
    return result.sort_values(["datetime", "uid"]).reset_index(drop=True)


def wake_up_times(X):
    """First non-sleeping event of each person (their wake-up), as time of day."""
    first = X[X["state"] != "Sleeping"].sort_values("datetime").groupby("uid").first()
    return first["datetime"].dt.strftime("%H:%M")


def state_summary(X):
    """People, rows and hours per country and state (rows include size-only rows)."""
    X = X.sort_values(["uid", "datetime"]).copy()
    X["hours"] = (X.groupby("uid")["datetime"].shift(-1) - X["datetime"]).dt.total_seconds() / 3600
    return (
        X.groupby(["country", "state"])
        .agg(people=("uid", "nunique"), rows=("uid", "size"), hours=("hours", "sum"))
        .round(0)
    )


def model_report(models, countries=None):
    """Mean participation (%) and mean minutes per activity, per country. Spots missing models."""
    rows = []
    for c in (countries or sorted(models)):
        for a in sorted(set(models[c]["time"]) | set(models[c]["percentage"])):
            t, p = models[c]["time"].get(a), models[c]["percentage"].get(a)
            rows.append({
                "country": c, "activity": a,
                "participation_%": None if p is None else round(float(np.mean(p.generate(500))), 1),
                "minutes": None if t is None else round(float(np.mean(t.generate(500))), 0),
            })
    return pd.DataFrame(rows)


def validate_log(X, base_size=1.0, eating_size=2.0):
    assert X.columns.tolist() == ["datetime", "uid", "state", "country", "size"]
    assert X["datetime"].is_monotonic_increasing
    assert not X.duplicated(["datetime", "uid"]).any()
    # Each person belongs to exactly one country.
    assert (X.groupby("uid")["country"].nunique() == 1).all()
    # Size stays within its bounds.
    lo, hi = min(base_size, eating_size), max(base_size, eating_size)
    assert X["size"].between(lo, hi).all()
    # A row repeating the previous state must exist because the size changed.
    g = X.groupby("uid")
    same_state = X["state"] == g["state"].shift()
    same_size = X["size"] == g["size"].shift()
    assert not (same_state & same_size).any()


if __name__ == "__main__":
    from life_of_eurpean import load_time_use, fit_models

    models = fit_models(load_time_use())

    X = simulate_activity_log(
        models=models,
        countries=["Netherlands", "Germany", "Finland", "Poland"],
        k=100,
        start_date="2000-01-01",
        seed=42,
    )
    validate_log(X)

    print(X)
    print(X[X["uid"] == 0])
    print(X.groupby("country")["uid"].nunique())
    print(X.groupby("country")["state"].value_counts().unstack(0))
