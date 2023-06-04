from Bio import SeqIO
def search_vcf_by_rsid(vcf_file, rsid):
    vcf_records = []

    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            record_rsid = fields[2]
            if record_rsid == rsid:
                chromosome = fields[0]
                pos = fields[1]
                reference_allele = fields[3]
                alternate_allele = fields[4]
                vcf_records.append((chromosome,pos,reference_allele, alternate_allele,record_rsid))
                record = list(vcf_records[0])
                record[0] = 'chr' + record[0]
                break  # Stop searching after finding a match
    return record
from pyfaidx import Fasta

def compare_fasta_by_name(fasta_file,data_list):
    # 读取FASTA文件
    final_query = dict()
    records = list(SeqIO.parse(fasta_file, "fasta"))
    # 遍历记录并打印每个记录的名称和序列
    for record in records:
        name = record.id
        sequence = record.seq
        if name == data_list[0]:
            final_query['CHROM'] = name
            final_query['Reference Allele'] = data_list[2]
            final_query['Alternate Allele'] = data_list[3]
            sequence_first,sequence_last = extract_upstream_downstream(data_list[1],sequence,20)
            final_query['upstream_sequence'] = sequence_first
            final_query['downstream_sequence'] = sequence_last
            final_query['rsID'] = data_list[4]
            break
    return final_query
def extract_upstream_downstream(position, sequence, N):
    # 确保N在有效范围内
    position = int(position)
    N = max(min(N, 100), 1)
    # 确定上游和下游的起始位置
    upstream_start = max(position - N, 0)
    downstream_end = position + N

    # 提取上游和下游的基因组序列
    upstream_sequence = sequence[upstream_start:position]
    downstream_sequence = sequence[position:downstream_end]

    return upstream_sequence, downstream_sequence

def dict_to_vcf(dictionary, output_file):
    # 打开输出文件以写入 VCF 数据
    with open(output_file, 'w') as file:
        # 写入 VCF 文件头部信息
        file.write('##fileformat=VCFv4.3\n')
        file.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        file.write('#CHROM\tReference Allele\tAlternate Allele\tupstream_sequence\tdownstream_sequence\trsid\n')

        # 提取字典中的字段值
        chrom = dictionary.get('CHROM', '.')
        ref = dictionary.get('Reference Allele', '.')
        alt = dictionary.get('Alternate Allele', '.')
        upstream_seq = str(dictionary.get('upstream_sequence', ''))
        downstream_seq = str(dictionary.get('downstream_sequence', ''))
        rsid = dictionary.get('rsID', '.')

        # 将字段值按照 VCF 格式写入输出文件
        file.write(f'{chrom}\t{ref}\t{alt}\t{upstream_seq}\t{downstream_seq}\t{rsid}\n')


def final_result(rsid,vcf_file, fasta_file,outputfile):
    data_list = search_vcf_by_rsid(vcf_file, rsid)
    print(data_list)
    test_dict = compare_fasta_by_name(fasta_file, data_list)
    return dict_to_vcf(test_dict,outputfile)
