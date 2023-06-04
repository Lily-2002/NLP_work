import click
from shell_project.find import final_result


@click.command()
@click.argument('rsid', type=str)
@click.argument('vcf_file', type=click.Path(exists=True))
@click.argument('fasta_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path(exists=True))
def cli(rsid,vcf_file, fasta_file,out_file):
    # 调用 find.py 中的 final_result 函数，并传入 RSID 参数
    result  = final_result(rsid,vcf_file, fasta_file,out_file)
    # 处理结果
    if result is not None:
        click.echo(f"找到了与 RSID '{rsid}' 相关的结果:")
        # 输出结果
        print(result)
    else:
        click.echo(f"找不到与 RSID '{rsid}' 相关的结果.")


if __name__ == '__main__':
    cli()
