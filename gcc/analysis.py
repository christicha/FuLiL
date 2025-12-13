from configparser import ConfigParser
import os


def analysis(bugid, revision, configPath):
    cfg = ConfigParser()
    cfg.read(configPath)
    infoResultDir = cfg.get('gcc-locations', 'infodir')
    location = infoResultDir + '/' + bugid + '/locations'
    baseResultDir = cfg.get('gcc-locations', 'resultFile')
    logBaseDir = cfg.get('gcc-locations', 'logDir')
    GNNfile = baseResultDir + bugid + '/result_gnn.csv'
    SBFLfile = baseResultDir + bugid + '/resultFile_file.csv'
    LLMfile = logBaseDir + bugid + '/result_llm.csv'
    aggregatefile = baseResultDir + bugid + '/aggregate_result.csv'
    analysis_csv = os.path.join(baseResultDir, bugid, 'analysis.csv')

    buggy_names = []
    with open(location, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        buggy_keywords = ['buggy location', 'buggy locations', 'buggy locations:', 'buggy location:']
        buggy_idx = -1
        for keyword in buggy_keywords:
            if keyword in lines:
                buggy_idx = lines.index(keyword)
                break
        if buggy_idx == -1:
            print("未找到")
            return

        for line in lines[buggy_idx + 1:]:
            if line.startswith('file:'):
                full_path = line.split('file:')[1].split(';')[0]
                buggy_name = full_path.replace('gcc/', '')
                buggy_name = buggy_name.replace('trunk/', '')
                # 处理开头的/
                buggy_name = buggy_name.lstrip()
                if buggy_name.startswith('/'):
                    buggy_name = buggy_name[1:]
                if buggy_name not in buggy_names:
                    buggy_names.append(buggy_name)

    def get_rank(file_path, col_name):
        if not os.path.exists(file_path):
            return 'N/A'
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            if col_name not in header:
                return 'N/A'
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1] == buggy_name:
                    return parts[0]
        return 'N/A'

    os.makedirs(os.path.dirname(analysis_csv), exist_ok=True)

    with open(analysis_csv, 'a', encoding='utf-8') as f:
        if os.path.getsize(analysis_csv) == 0:
            f.write("File,SBFL,GNN,LLM,Aggregate\n")
        for buggy_name in buggy_names:
            sbfl_rank = get_rank(SBFLfile, 'File')
            gnn_rank = get_rank(GNNfile, 'File')
            llm_rank = get_rank(LLMfile, 'Filename')
            aggregate_rank = get_rank(aggregatefile, 'File')
            write_line = f"{buggy_name},{sbfl_rank},{gnn_rank},{llm_rank},{aggregate_rank}\n"
            f.write(write_line)
            print(
                f"bugid:{bugid} | 结果：{buggy_name} | SBFL排名：{sbfl_rank} | GNN排名：{gnn_rank} | LLM排名：{llm_rank} | Aggregate排名：{aggregate_rank}")


if __name__ == "__main__":
    res = analysis("15920", "r181995", "/home/chris/FLL/config/config.ini")