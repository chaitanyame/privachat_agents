"""add_excluded_domains_table

Revision ID: 003
Revises: 002
Create Date: 2025-11-17 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create excluded_domains table for domain exclusion management.
    
    This migration:
    1. Creates excluded_domains table with domain patterns and types
    2. Adds indexes for performance (is_active, domain_pattern)
    3. Creates trigger to auto-update updated_at timestamp
    4. Inserts initial YouTube domain exclusions using wildcard patterns
    """
    
    # Create excluded_domains table
    op.create_table(
        "excluded_domains",
        sa.Column("id", UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("domain_pattern", sa.String(length=255), nullable=False),
        sa.Column("pattern_type", sa.String(length=20), nullable=False, server_default="exact"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("created_by", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("domain_pattern", name="uq_excluded_domains_domain_pattern"),
        sa.CheckConstraint(
            "pattern_type IN ('exact', 'wildcard', 'regex')",
            name="check_pattern_type"
        ),
    )
    
    # Create indexes for performance
    op.create_index(
        "idx_excluded_domains_is_active",
        "excluded_domains",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        "idx_excluded_domains_domain_pattern",
        "excluded_domains",
        ["domain_pattern"],
        unique=False,
    )
    
    # Create trigger function for updating updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_excluded_domains_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger
    op.execute("""
        CREATE TRIGGER trigger_update_excluded_domains_updated_at
        BEFORE UPDATE ON excluded_domains
        FOR EACH ROW
        EXECUTE FUNCTION update_excluded_domains_updated_at();
    """)
    
    # Insert initial YouTube domain exclusions using wildcard patterns
    # *.youtube.* matches all YouTube subdomains (www, m, music, etc.) across all TLDs
    op.execute("""
        INSERT INTO excluded_domains (domain_pattern, pattern_type, reason, created_by)
        VALUES 
            ('*.youtube.*', 'wildcard', 'Video platform with subdomains - crawling not useful for text search', 'system'),
            ('youtube.com', 'exact', 'Video platform main domain - crawling not useful for text search', 'system'),
            ('youtu.be', 'exact', 'YouTube short links - video platform', 'system');
    """)


def downgrade() -> None:
    """Drop excluded_domains table and related triggers."""
    
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_excluded_domains_updated_at ON excluded_domains")
    op.execute("DROP FUNCTION IF EXISTS update_excluded_domains_updated_at()")
    
    # Drop indexes
    op.drop_index("idx_excluded_domains_domain_pattern", table_name="excluded_domains")
    op.drop_index("idx_excluded_domains_is_active", table_name="excluded_domains")
    
    # Drop table
    op.drop_table("excluded_domains")
