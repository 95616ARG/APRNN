#! /usr/bin/env python3

import os, subprocess, argparse, pathlib
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--eval', '-e', type=int, dest='eval', action='store', required=True,
                    choices=[1,2,3,4,5,6,7,8],
                    help='The experiment to run.')
parser.add_argument('--tool', '-t', type=str, dest='tool', action='store',
                    default='all',
                    choices=['aprnn', 'prdnn', 'lookup', 'all'],
                    help='Tools to run in experiments (if applicable).')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='all',
                    choices=['all', '3x100', '9x100', '9x200', 'resnet152', 'vgg19', 'n29'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' artifact to reproduce numbers in paper.')
parser.add_argument('--rerun', dest='rerun', action='store_true',
                    help='Force to rerun experiments.')
parser.add_argument('--norun', dest='norun', action='store_true',
                    help='Force to not run experiments, just report results from previous runs.')
parser.add_argument('--npoints', dest='npoints', type=str,
                    help='"all" or number of repair points (may not be applicable for all experiments).')


def check_networks(eval, net, valid_nets):
    if net not in valid_nets:
        raise RuntimeError(
            f"Invalid network '{net}' for Experiment {eval}; please choose from {valid_nets}."
        )
    if net == 'all':
        return tuple(n for n in valid_nets if n != 'all')
    else:
        return tuple((net,))

def check_tools(eval, tool, valid_tools):
    if tool not in valid_tools:
        raise RuntimeError(
            f"Invalid network '{tool}' for Experiment {eval}; please choose from {valid_tools}."
        )
    if tool == 'all':
        return tuple(t for t in valid_tools if t != 'all')
    else:
        return tuple((tool,))

def args_to_string(args, excludes=['eval', 'net', 'tool', 'rerun', 'norun', 'npoints']):
    cmds = []
    for k, v in vars(args).items():
        if k in excludes:
            continue
        if isinstance(v, bool):
            if v == True:
                cmds.append(f"--{k}")
        else:
            cmds.append(f"--{k}='{v}'")

    return ' '.join(cmds)

def run_command(cmd):
    print(cmd)
    return subprocess.run(cmd, shell=True, check=False)

results_root = pathlib.Path('results')

def get_result(eval, tool, net, use_artifact, suffix=""):
    prefix = 'artifact_' if use_artifact else ''
    return results_root / f'eval_{eval}' / f'{prefix}{tool}_{net}{suffix}.npy'

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

def msg(s):
    print_msg_box(s)

def load_result(result, *args, **kwargs):
    # print(f"Reading result from {result.as_posix()}")
    return np.load(result, *args, **kwargs)

def common_msgs(args):

    if args.use_artifact:
        msg("Note: Evaluated using authors' artifact because of `--use_artifact`.")
    else:
        # msg(
        #     "Note: Ran experiments on this machine and evaluated results. \n"
        #     "Note that the results, especially timing numbers might be \n"
        #     "different because of difference in CPU, GPU, memory and Gurobi's\n"
        #     "non-determinism on different machines/runs. "
        # )
        pass


if __name__ == '__main__':
    args = parser.parse_args()

    if args.eval == 7:
        msg("Please run `eval_7_aprnn.py`.")
        exit(0)

    other_args = args_to_string(args)

    if args.norun:
        msg("Will not run experiments because of `--norun`; will report results from priori runs.")

    elif args.rerun:
        msg("Will re-run experiments because of `--rerun`.")

    if args.eval == 1:
        nets = check_networks(args.eval, args.net, ('all', '3x100', '9x100', '9x200'))
        tools = check_tools(args.eval, args.tool, ('all', 'prdnn', 'aprnn', 'lookup', 'reassure'))

        for tool in tools:
            for net in nets:
                if args.norun:
                    msg( "Not atually running experiments because of `--norun`." )
                    pass
                elif args.rerun or not get_result(args.eval, tool, net, args.use_artifact).exists():
                    run_command(
                        f"python3 ./eval_{args.eval}_{tool}.py --net={net} {other_args}"
                    )
                else:
                    msg(
                        """Reusing cached result from previous runs
{cache_path}
To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
    cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
)
                    )


        df = None
        for tool in tools:
            tool_result_dict = {}
            for net in nets:
                result = get_result(args.eval, tool, net, args.use_artifact)
                if result.exists():
                    tool_result_dict[net] = load_result(result, allow_pickle=True).item()[net]
                else:
                    tool_result_dict[net] = {
                        (tool.upper(), 'D'): '(to run)',
                        (tool.upper(), 'G'): '(to run)',
                        (tool.upper(), 'T'): '(to run)',
                    }

            df_tool = pd.DataFrame.from_dict(tool_result_dict, orient='index')
            if df is None:
                df = df_tool
            else:
                df = df.join(df_tool)

        common_msgs(args)

        msg(
            """Results corresponds to Table 1,
for super-columns {tools} and rows {nets}:

            """.format(tools=tuple(t.upper() for t in tools), nets=nets) +
            df.to_string(
                index=True,
                justify='center',
                float_format='{:.2%}'.format,
                col_space=8,
                na_rep='(*)',
            ) +
"""

Metrics:
- D for drawdown, lower is better.
- G for generalization, higher is better.
- T for time. """
        )

    elif args.eval == 2:
        nets = check_networks(args.eval, args.net, ('all', 'resnet152', 'vgg19'))
        tools = check_tools(args.eval, args.tool, ('all', 'aprnn', 'prdnn'))

        for tool in tools:
            for net in nets:
                if args.norun:
                    pass
                elif args.rerun or not get_result(args.eval, tool, net, args.use_artifact).exists():
                    run_command(
                        f"python3 ./eval_{args.eval}_{tool}.py --net={net} {other_args}"
                    )
                else:
                    msg(
                        """Reusing cached result from previous runs
{cache_path}
To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
    cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
)
                    )

        df = None
        result_dict = {}
        for tool in tools:
            df_tool = None
            result_dict[tool.upper()] = {}
            for net in nets:
                result = get_result(args.eval, tool, net, args.use_artifact)
                if result.exists():
                    result_dict[tool.upper()]= np.load(result, allow_pickle=True).item()[tool.upper()]
                else:
                    if not args.norun:
                        result_dict[tool.upper()] = {
                            (net, 'D@top-1'): '(failed)',
                            (net, 'D@top-5'): '(failed)',
                            (net, 'T'): '(failed)',
                        }
                        # np.save(result.as_posix(), {tool.upper(): result_dict[tool.upper()]}, allow_pickle=True)
                    else:
                        result_dict[tool.upper()] = {
                            (net, 'D@top-1'): '(to run)',
                            (net, 'D@top-5'): '(to run)',
                            (net, 'T'): '(to run)',
                        }

                df_tool_net = pd.DataFrame.from_dict({tool.upper(): result_dict[tool.upper()]}, orient='index')
                if df_tool is None:
                    df_tool = df_tool_net
                else:
                    df_tool = df_tool.join(df_tool_net)

            if df is None:
                df = df_tool
            else:
                df = pd.concat((df, df_tool))

        msg("""Results corresponds to Section 6.2 on page 16 (lines 760-766),
for specified tools {tools} and networks {nets}:

{table}

Metrics:
- D@top-1 for top-1 accuracy drawdown, lower is better.
- D@top-5 for top-5 accuracy drawdown, lower is better.
- T for time, lower is better. """.format(
                tools = tools,
                nets = nets,
                table = df.to_string(
                    index=True,
                    justify='center',
                    float_format='{:.2%}'.format,
                    col_space=8,
                    na_rep='(*)',
                )
            )
        )

    elif args.eval == 3:
        nets = check_networks(args.eval, args.net, ('all', '9x100'))
        tools = check_tools(args.eval, args.tool, ('all', 'aprnn', 'prdnn', 'lookup', 'reassure'))

        for tool in tools:
            for net in nets:
                if args.norun:
                    pass
                elif args.rerun or not get_result(args.eval, tool, net, args.use_artifact).exists():
                    run_command(
                        f"python3 ./eval_{args.eval}_{tool}.py --net={net} {other_args}"
                    )
                else:
                    msg(
                        """Reusing cached result from previous runs
{cache_path}
To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
    cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
)
                    )

        net = nets[0]
        result_dict = {}
        for tool in tools:
            result = get_result(args.eval, tool, net, args.use_artifact)
            if result.exists():
                result_dict[tool.upper()] = load_result(result, allow_pickle=True).item()[tool.upper()]
            else:
                result_dict[tool.upper()] = {
                    'D': '(to run)',
                    'G1': '(to run)',
                    'G2': '(to run)',
                    'T': '(to run)',
                }

        common_msgs(args)

        if args.tool == 'aprnn':
            result_string = """
Regarding time (line 818):
    This work (APRNN) took {T_aprnn} seconds.

Regarding drawdown (lines 820-821):
    This work (APRNN)'s drawdown is {D_aprnn}.
Remark: This work (ARPNN) has a good (low) drawdown. The authors' repaired network (`--use_artifact`)
has a negative drawdown.

Regarding generalization on generalization set S1 (lines 822-824):
    This work (APRNN)'s generalization is {G1_aprnn}.
Remark: This work (ARPNN) has a good (high) generalization on generalization set S1.

Regarding generalization on generalization set S2 (lines 824-827):
    This work (APRNN)'s generalization is {G2_aprnn}.
Remark: This work (ARPNN) has a good (high) generalization on generalization set S2.
            """.format(
                T_aprnn = result_dict['APRNN']['T'],
                D_aprnn = result_dict['APRNN']['D'],
                G1_aprnn = result_dict['APRNN']['G1'],
                G2_aprnn = result_dict['APRNN']['G2'],
            )

        elif args.tool == 'prdnn':
            result_string = """
Regarding time (line 818):
    The baseline (PRDNN) took {T_prdnn} seconds.

Regarding drawdown (lines 820-821):
    The baseline (PRDNN)'s drawdown is {D_prdnn}.
Remark: The baseline (PRDNN) has a good (low) drawdown. The authors' repaired network (`--use_artifact`)
has a negative drawdown.

Regarding generalization on generalization set S1 (lines 822-824):
    The baseline (PRDNN)'s generalization is {G1_prdnn}.
Remark: The baseline (PRDNN) has a good (high) generalization on generalization set S1.

Regarding generalization on generalization set S2 (lines 824-827):
    The baseline (PRDNN)'s generalization is {G2_prdnn}.
Remark: The baseline (PRDNN) has a good (high) generalization on generalization set S2.
            """.format(
                T_prdnn  = result_dict['PRDNN']['T'],
                D_prdnn  = result_dict['PRDNN']['D'],
                G1_prdnn = result_dict['PRDNN']['G1'],
                G2_prdnn = result_dict['PRDNN']['G2'],
            )

        elif args.tool == 'all':
            result_string = """
Regarding time (line 818):
    This work    (APRNN) took {T_aprnn} seconds;
    The baseline (PRDNN) took {T_prdnn} seconds.

Regarding drawdown (lines 820-821):
    This work    (APRNN)'s drawdown is {D_aprnn};
    The baseline (PRDNN)'s drawdown is {D_prdnn}.
Remark: Both tools has a good (low) drawdown. The authors' repaired networks (`--use_artifact`)
have negative drawdown for both tools.

Regarding generalization on generalization set S1 (lines 822-824):
    This work    (APRNN)'s generalization is {G1_aprnn};
    The baseline (PRDNN)'s generalization is {G1_prdnn}.
Remark: Both tools has a good (high) generalization on generalization set S1.

Regarding generalization on generalization set S2 (lines 824-827):
    This work    (APRNN)'s generalization is {G2_aprnn};
    The baseline (PRDNN)'s generalization is {G2_prdnn}.
Remark: Both tools has a good (high) generalization on generalization set S2.
            """.format(
                T_aprnn  = result_dict['APRNN']['T'],
                D_aprnn  = result_dict['APRNN']['D'],
                G1_aprnn = result_dict['APRNN']['G1'],
                G2_aprnn = result_dict['APRNN']['G2'],
                T_prdnn  = result_dict['PRDNN']['T'],
                D_prdnn  = result_dict['PRDNN']['D'],
                G1_prdnn = result_dict['PRDNN']['G1'],
                G2_prdnn = result_dict['PRDNN']['G2'],
            )

        msg(
            """Results corresponds to Section 6.3 on page 17 (lines 818-827) for the specified tools {tools}:
            """.format(tools=tools) + result_string
        )

    elif args.eval == 4:
        nets = check_networks(args.eval, args.net, ('all', 'n29'))
        tools = check_tools(args.eval, args.tool, ('all', 'aprnn', 'prdnn'))

        for tool in tools:
            for net in nets:
                if args.norun:
                    pass

                elif args.rerun or not get_result(args.eval, tool, net, args.use_artifact).exists():
                    run_command(
                        f"python3 ./eval_{args.eval}_{tool}.py --net={net} {other_args}"
                    )
                else:
                    msg(
                        """Reusing cached result from previous runs
{cache_path}
To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
    cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
)
                    )

        net = nets[0]
        result_dict = {}
        for tool in tools:
            result = get_result(args.eval, tool, net, args.use_artifact)
            if result.exists():
                result_dict[tool.upper()] = load_result(result, allow_pickle=True).item()[tool.upper()]
            else:
                result_dict[tool.upper()] = {
                    'D': '(to run)',
                    'G': '(to run)',
                    'T': '(to run)',
                }

        common_msgs(args)

        if args.tool == 'aprnn':

            msg(
                """Results corresponds to Section 6.4 on page 18 (lines 862-865) for specified tools {tools}:

Regarding time (line 862):
    This work (APRNN) took {T_aprnn} seconds.

Regarding property drawdown (line 863):
    This work (APRNN)'s property drawdown is {D_aprnn}.
Remark: This work (ARPNN) has a good (low) property drawdown.

Regarding property generalization (line 864):
    This work (APRNN)'s property generalization is {G_aprnn}.
Remark: This work (ARPNN) has a good (high) property generalization. """.format(
                    tools=tools,
                    T_aprnn = result_dict['APRNN']['T'],
                    D_aprnn = result_dict['APRNN']['D'],
                    G_aprnn = result_dict['APRNN']['G'],
                    # T_prdnn = result_dict['PRDNN']['T'],
                    # D_prdnn = result_dict['PRDNN']['D'],
                    # G_prdnn = result_dict['PRDNN']['G'],
                )
            )

        elif args.tool == 'prdnn':
            msg(
                """Results corresponds to Section 6.4 on page 18 (lines 862-865) for specified tools {tools}:

Regarding time (line 862):
    The baseline (PRDNN) took {T_prdnn} seconds.

Regarding property drawdown (line 863):
    The baseline (PRDNN)'s property drawdown is {D_prdnn}.
Remark: The baseline (PRDNN) has a good (low) property drawdown.

Regarding property generalization (line 864):
    The baseline (PRDNN)'s property generalization is {G_prdnn}.
Remark: The baseline (PRDNN) has a good (high) property generalization. """.format(
                    tools=tools,
                    T_prdnn = result_dict['PRDNN']['T'],
                    D_prdnn = result_dict['PRDNN']['D'],
                    G_prdnn = result_dict['PRDNN']['G'],
                )
            )

        else:
            msg(
                """Results corresponds to Section 6.4 on page 18 (lines 862-865) for specified tools {tools}:

Regarding time (line 862):
    This work    (APRNN) took {T_aprnn} seconds;
    The baseline (PRDNN) took {T_prdnn} seconds.

Regarding property drawdown (line 863):
    This work    (APRNN)'s property drawdown is {D_aprnn};
    The baseline (PRDNN)'s property drawdown is {D_prdnn}.
Remark: Both tools have a good (low) property drawdown.

Regarding property generalization (line 864):
    This work    (APRNN)'s property generalization is {G_aprnn};
    The baseline (PRDNN)'s property generalization is {G_prdnn}.
Remark: Both tools have a good (low) property generalization. """.format(
                    tools=tools,
                    T_aprnn = result_dict['APRNN']['T'],
                    D_aprnn = result_dict['APRNN']['D'],
                    G_aprnn = result_dict['APRNN']['G'],
                    T_prdnn = result_dict['PRDNN']['T'],
                    D_prdnn = result_dict['PRDNN']['D'],
                    G_prdnn = result_dict['PRDNN']['G'],
                )
            )

    elif args.eval == 5:
        nets = check_networks(args.eval, args.net, ('all', 'n29'))
        tools = check_tools(args.eval, args.tool, ('all', 'aprnn'))

        for tool in tools:
            for net in nets:
                if args.norun:
                    pass
                elif args.rerun or not get_result(args.eval, tool, net, args.use_artifact).exists():
                    run_command(
                        f"python3 ./eval_{args.eval}_{tool}.py --net={net} {other_args}"
                    )
                else:
                    msg(
                        """Reusing cached result from previous runs
{cache_path}
To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
    cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
)
                    )

        net = nets[0]
        result_dict = {}
        for tool in tools:
            result = get_result(args.eval, tool, net, args.use_artifact)
            if result.exists():
                result_dict[tool.upper()] = load_result(result, allow_pickle=True).item()[tool.upper()]
            else:
                result_dict[tool.upper()] = {
                    'T': '(to run)',
                }

        common_msgs(args)

        msg(
            """Results corresponds to Section 6.5 on page 18 (line 878) for APRNN:

The work (ARPNN) took {T} seconds. """.format(
                tools=tools,
                T = result_dict['APRNN']['T']
            )
        )

    elif args.eval == 6:
        nets = check_networks(args.eval, args.net, ('all', '3x100_gelu', '3x100_hardswish'))
        tools = check_tools(args.eval, args.tool, ('all', 'aprnn'))

        if args.npoints == 'all':
            args.npoints = (1, 10, 50, 100)
        else:
            args.npoints = (int(args.npoints),)
        
        for points in args.npoints:
            suffix=f"_{points}"
            for tool in tools:
                for net in nets:
                    if args.norun:
                        msg( "Not atually running experiments because of `--norun`." )
                        pass
                    elif args.rerun or not get_result(
                        args.eval, tool, net, args.use_artifact, suffix=suffix).exists():
                        run_command(
                            f"python3 ./eval_{args.eval}_{tool}.py --net={net} --npoints {points} {other_args}"
                        )
                    else:
                        msg(
                            """Reusing cached result from previous runs
    {cache_path}
    To discard cache and re-run the specified experiment, append the option `--rerun`. """.format(
        cache_path = get_result(args.eval, tool, net, args.use_artifact).as_posix()
    )
                        )


        df = None
        for tool in tools:
            tool_result_dict = {}
            for net in nets:
                for points in args.npoints:
                    suffix=f"_{points}"
                    result = get_result(args.eval, tool, net, args.use_artifact, suffix=suffix)
                    if result.exists():
                        tool_result_dict[points] = load_result(result, allow_pickle=True).item()[str(points)]
                    else:
                        tool_result_dict[points] = {
                            (net, 'D'): '(to run)',
                            (net, 'G'): '(to run)',
                            (net, 'T'): '(to run)',
                        }

                df_tool = pd.DataFrame.from_dict(tool_result_dict, orient='index')
                if df is None:
                    df = df_tool
                else:
                    df = df.join(df_tool)

        common_msgs(args)

        msg(
            """Results corresponds to Table 2,
for super-columns {nets} and rows {points}:

            """.format(nets=nets, points=args.npoints) +
            df.to_string(
                index=True,
                justify='center',
                float_format='{:.2%}'.format,
                col_space=8,
                na_rep='(*)',
            ) +
"""

Metrics:
- D for drawdown, lower is better.
- G for generalization, higher is better.
- T for time. """
        )


    else:
        raise RuntimeError(
            f"Invalid experiment {args.eval}; please specify 1, 2, 3, 4 or 5."
        )
