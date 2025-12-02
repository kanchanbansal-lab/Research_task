import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plot_flight_altitude_speed(
    d,
    customdata,
    phase_colors,
    flight_id,
    show=True,
):
    """
    Plot altitude, vertical speed, horizontal speed and phase bands.

    Parameters
    ----------
    d : pandas.DataFrame
    customdata : array-like
        Customdata for hover (shape: (n, 4) -> altitude, vz, hz, phase).
    phase_colors : dict
        Mapping {phase_name: color_string}.
    flight_id : str or int
        Identifier for the flight, used in the title.
    show : bool, default True
        Whether to immediately display the figure.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    y_min = d["altitude_measured"].min() - 1
    y_max = d["altitude_measured"].max() + 1

    fig = go.Figure()

    # ------------------ 1) Add Altitude (left axis) ------------------
    fig.add_trace(go.Scatter(
        x=d["time"],
        y=d["altitude_measured"],
        mode="lines",
        line=dict(width=2, color="blue"),
        name="Altitude (m)",
        customdata=customdata,
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Altitude: %{customdata[0]:.2f} m<br>"
            "Vertical speed: %{customdata[1]:.2f} m/s<br>"
            "Horizontal speed: %{customdata[2]:.2f} m/s<br>"
            "Phase: %{customdata[3]}<extra></extra>"
        )
    ))

    # ------------------ 2) Add Vertical Speed (right axis) ------------------
    fig.add_trace(go.Scatter(
        x=d["time"],
        y=d["vz_from_alt"],
        mode="lines",
        line=dict(width=2, color="red"),
        name="Vertical speed (m/s)",
        yaxis="y2",
        customdata=customdata,
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Altitude: %{customdata[0]:.2f} m<br>"
            "Vertical speed: %{customdata[1]:.2f} m/s<br>"
            "Horizontal speed: %{customdata[2]:.2f} m/s<br>"
            "Phase: %{customdata[3]}<extra></extra>"
        )
    ))

    # ------------------ 2b) Add Horizontal Speed (right axis) ------------------
    fig.add_trace(go.Scatter(
        x=d["time"],
        y=d["horizontal_speed"],
        mode="lines",
        line=dict(width=2, color="green"),
        name="Horizontal speed (m/s)",
        yaxis="y2",
        customdata=customdata,
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Altitude: %{customdata[0]:.2f} m<br>"
            "Vertical speed: %{customdata[1]:.2f} m/s<br>"
            "Horizontal speed: %{customdata[2]:.2f} m/s<br>"
            "Phase: %{customdata[3]}<extra></extra>"
        )
    ))

    # ------------------ 3) Add Phase Bands as Filled Traces ------------------
    # work on a sorted copy to avoid mutating original df
    df = d.sort_values("time").reset_index(drop=True)

    # prepare containers: one x/y list per phase
    bands = {
        phase: {"x": [], "y": []}
        for phase in phase_colors.keys()
    }

    # iterate over each *interval* [time[i], time[i+1]]
    for i in range(len(df) - 1):
        phase = df.loc[i, "phase"]
        if phase not in phase_colors:
            continue  # skip phases you don't want to paint

        t0 = df.loc[i, "time"]
        t1 = df.loc[i + 1, "time"]

        # build a small rectangle for this interval
        bands[phase]["x"] += [t0, t1, t1, t0, t0, None]
        bands[phase]["y"] += [y_min, y_min, y_max, y_max, y_min, None]

    # now add one trace per phase
    for phase, color in phase_colors.items():
        x_band = bands[phase]["x"]
        y_band = bands[phase]["y"]
        if not x_band:
            continue

        fig.add_trace(go.Scatter(
            x=x_band,
            y=y_band,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            name=f"Phase: {phase}",
            hoverinfo="skip",
            showlegend=True
        ))

    # ------------------ 4) Layout ------------------
    fig.update_layout(
        title=f"Flight {flight_id} – Altitude, Vertical Speed & Phases",
        xaxis=dict(title="Time (s)"),

        # Left Y-axis
        yaxis=dict(
            title=dict(text="Altitude (m)", font=dict(color="blue")),
            tickfont=dict(color="blue"),
            range=[y_min, y_max]
        ),

        # Right Y-axis
        yaxis2=dict(
            title=dict(text="Vertical speed (m/s)", font=dict(color="red")),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right"
        ),

        template="plotly_white",
        legend=dict(x=1, y=1)
    )

    fig.update_layout(
        legend=dict(
            x=1.08,
            y=1,
            bgcolor="rgba(255,255,255,0.7)"
        ),
        margin=dict(r=120)
    )

    if show:
        fig.show()

    return fig

def compute_airspeed(df, wind_x, wind_y):
    return np.sqrt(
        (df["velocity_x"] - wind_x)**2 +
        (df["velocity_y"] - wind_y)**2
    )

def MAE(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    """
    Root Mean Square Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error (in %)
    Avoids division-by-zero by adding a very small epsilon.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# --------------------------------
# Main cleaning function
# --------------------------------

MAX_WIND_SPEED = 60.0          # m/s
MIN_BATTERY_VOLT = 5.0         # V
MAX_BATTERY_VOLT = 30.0        # V
MAX_BATTERY_CURRENT = 500.0    # A (absolute)

MAX_POS_ABS = 1e6              # m, sanity bound on |x|,|y|,|z|
MAX_SPEED_FROM_POS = 100.0     # m/s, implied speed from position
MAX_VEL = 50.0                 # m/s, |velocity_*|
MAX_SPEED = 100.0              # m/s, |speed|

MAX_ANGULAR_RATE = 50.0        # rad/s
MAX_ACCEL = 100.0              # m/s^2

MIN_ALTITUDE = -100.0          # m
MAX_ALTITUDE = 5000.0          # m

# how far quaternion norm can deviate from 1 and still be renormalized
QUAT_NORM_MIN = 0.9
QUAT_NORM_MAX = 1.1

def clean_flight_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean telemetry dataframe in-place-like, returning a new cleaned copy.
    """
    df = df.copy()

    # -----------------------------
    # 1. Basic type cleaning
    # -----------------------------

    # flight / route as strings
    if 'flight' in df.columns:
        df['flight'] = pd.to_numeric(df['flight'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['flight'])

    if 'route' in df.columns:
        df['route'] = df['route'].astype(str).str.strip()

    # date + time_day → timestamp
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    if 'time_day' in df.columns:
        # try common formats – adjust format if needed
        df['time_day'] = pd.to_datetime(df['time_day'], errors='coerce').dt.time

    if {'date', 'time_day'}.issubset(df.columns):
        df['timestamp'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time_day'].astype(str),
            errors='coerce'
        )
    else:
        raise ValueError("Need either ['date','time_day'] or 'time' to build a timestamp.")

    # Drop rows with no valid timestamp
    df = df.dropna(subset=['timestamp'])

    # time (relative) as numeric if present
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # -----------------------------
    # 2. Sort, deduplicate, time consistency
    # -----------------------------

    sort_keys = ['flight', 'timestamp'] if 'flight' in df.columns else ['timestamp']
    df = df.sort_values(sort_keys)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # # Drop rows where timestamp goes backwards within a flight
    # if 'flight' in df.columns:
    #     cummax = df.groupby('flight')['timestamp'].cummax()
    # else:
    #     cummax = df['timestamp'].cummax()

    # df = df[df['timestamp'] == cummax]
    # df = df.drop(columns=['timestamp'], errors='ignore')  # optionally keep; here we drop

    # # Recreate timestamp for further use (if we dropped above)
    # if {'date', 'time_day'}.issubset(df.columns):
    #     df['timestamp'] = pd.to_datetime(
    #         df['date'].astype(str) + ' ' + df['time_day'].astype(str),
    #         errors='coerce'
    #     )

    # -----------------------------
    # 3. Numeric casting
    # -----------------------------

    numeric_cols = [
        'wind_speed', 'wind_angle',
        'battery_voltage', 'battery_current',
        'position_x', 'position_y', 'position_z',
        'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
        'velocity_x', 'velocity_y', 'velocity_z',
        'angular_x', 'angular_y', 'angular_z',
        'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
        'speed', 'payload', 'altitude', 'time'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # -----------------------------
    # 4. Column-wise rules
    # -----------------------------

    # ---- wind_speed & wind_angle ----
    if 'wind_speed' in df.columns:
        df.loc[(df['wind_speed'] < 0) | (df['wind_speed'] > MAX_WIND_SPEED), 'wind_speed'] = np.nan

    # ---- battery_voltage & battery_current ----
    if 'battery_voltage' in df.columns:
        bad_v = (df['battery_voltage'] < MIN_BATTERY_VOLT) | (df['battery_voltage'] > MAX_BATTERY_VOLT)
        df.loc[bad_v, 'battery_voltage'] = np.nan

    if 'battery_current' in df.columns:
        bad_i = df['battery_current'].abs() > MAX_BATTERY_CURRENT
        df.loc[bad_i, 'battery_current'] = np.nan

    # ---- position_x, position_y, position_z ----
    pos_cols = ['position_x', 'position_y', 'position_z']
    pos_cols = [c for c in pos_cols if c in df.columns]

    if pos_cols:
        # sanity bound
        for c in pos_cols:
            df.loc[df[c].abs() > MAX_POS_ABS, c] = np.nan

        # compute implied speed from position differences per flight
        if 'timestamp' in df.columns:
            # midpoint: we'll drop rows where implied speed is insane
            if 'flight' in df.columns:
                group = df.groupby('flight', group_keys=False)
            else:
                group = [(None, df)]

            mask_bad_motion = pd.Series(False, index=df.index)

            for _, g in group:
                if len(g) < 2:
                    continue

                dt = g['timestamp'].diff().dt.total_seconds()
                dx = g['position_x'].diff() if 'position_x' in g else 0
                dy = g['position_y'].diff() if 'position_y' in g else 0
                dz = g['position_z'].diff() if 'position_z' in g else 0

                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                with np.errstate(divide='ignore', invalid='ignore'):
                    speed_impl = dist / dt

                bad_rows = (speed_impl > MAX_SPEED_FROM_POS) & dt.notna()
                mask_bad_motion.loc[g.index] |= bad_rows

            # set position values to NaN when motion is physically impossible
            for c in pos_cols:
                df.loc[mask_bad_motion, c] = np.nan

    # ---- quaternion: orientation_x,y,z,w ----
    quat_cols = ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
    quat_cols = [c for c in quat_cols if c in df.columns]

    if set(quat_cols) == set(['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']):
        q = df[quat_cols]
        norm = np.sqrt((q ** 2).sum(axis=1))

        good_norm = norm.between(QUAT_NORM_MIN, QUAT_NORM_MAX)
        # renormalize good ones
        df.loc[good_norm, quat_cols] = (q[good_norm].T / norm[good_norm]).T

        # bad ones to NaN
        df.loc[~good_norm, quat_cols] = np.nan

    # ---- velocity_x, velocity_y, velocity_z ----
    vel_cols = ['velocity_x', 'velocity_y', 'velocity_z']
    vel_cols = [c for c in vel_cols if c in df.columns]

    if vel_cols:
        for c in vel_cols:
            df.loc[df[c].abs() > MAX_VEL, c] = np.nan

    # ---- speed ----
    if 'speed' in df.columns:
        if set(vel_cols) == set(['velocity_x', 'velocity_y', 'velocity_z']):
            # recompute from velocity components
            df['speed_calc'] = np.sqrt(
                df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2
            )
            # replace speed with computed, but keep NaNs if velocity is NaN
            df['speed'] = df['speed_calc']
            df.drop(columns=['speed_calc'], inplace=True)

        # enforce range
        df.loc[df['speed'] < 0, 'speed'] = 0
        df.loc[df['speed'] > MAX_SPEED, 'speed'] = np.nan

    # ---- angular_x, angular_y, angular_z ----
    ang_cols = ['angular_x', 'angular_y', 'angular_z']
    ang_cols = [c for c in ang_cols if c in df.columns]

    if ang_cols:
        for c in ang_cols:
            df.loc[df[c].abs() > MAX_ANGULAR_RATE, c] = np.nan

    # ---- linear_acceleration_* ----
    acc_cols = [
        'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
    ]
    acc_cols = [c for c in acc_cols if c in df.columns]

    if acc_cols:
        for c in acc_cols:
            df.loc[df[c].abs() > MAX_ACCEL, c] = np.nan

    # ---- payload ----
    if 'payload' in df.columns:
        df.loc[df['payload'] < 0, 'payload'] = np.nan
        # payload assumed constant within a flight → ffill/bfill per flight
        if 'flight' in df.columns:
            df['payload'] = (
                df.groupby('flight')['payload']
                  .apply(lambda g: g.ffill().bfill())
                  .reset_index(level=0, drop=True)
            )

    # ---- altitude ----
    if 'altitude' in df.columns:
        bad_alt = (df['altitude'] < MIN_ALTITUDE) | (df['altitude'] > MAX_ALTITUDE)
        df.loc[bad_alt, 'altitude'] = np.nan

    # ---- route ----
    if 'route' in df.columns:
        # simple normalization: strip + uppercase (tweak as needed)
        df['route'] = df['route'].str.strip()
        df['route'] = df['route'].replace({'nan': np.nan, 'None': np.nan})
        df['route'] = df['route'].str.upper()

        if 'flight' in df.columns:
            df['route'] = (
                df.groupby('flight')['route']
                  .apply(lambda g: g.ffill().bfill())
                  .reset_index(level=0, drop=True)
            )

    # -----------------------------
    # 5. Interpolate numeric gaps within each flight
    # -----------------------------

    if 'flight' in df.columns:
        def _interp_group(g):
            # interpolate only numeric columns
            return g.interpolate(method='linear', limit_direction='both')

        df[numeric_cols] = (
            df.groupby('flight')[numeric_cols]
              .apply(_interp_group)
              .reset_index(level=0, drop=True)
        )
    else:
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    # -----------------------------
    # 6. Drop rows that are completely empty (all numeric NaN)
    # -----------------------------
    if numeric_cols:
        all_nan = df[numeric_cols].isna().all(axis=1)
        df = df[~all_nan]

    # Final sort
    df = df.sort_values(sort_keys).reset_index(drop=True)

    return df
